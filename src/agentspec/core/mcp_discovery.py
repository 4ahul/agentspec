"""MCP (Model Context Protocol) server discovery engine.

Discovers and parses MCP server tool definitions so agentspec can generate
tests for MCP tools.  MCP servers expose tools via the ``list_tools`` handler;
each tool has a *name*, *description*, and *inputSchema* (a JSON Schema object).

Two discovery strategies are supported:

1. **AST-based source parsing** -- walk a Python MCP server file looking for
   ``Tool(...)`` definitions inside ``@server.list_tools()`` handlers, as well
   as ``@server.call_tool()`` dispatcher patterns.
2. **JSON manifest parsing** -- read a static JSON file whose top-level
   ``tools`` key contains an array of MCP tool definitions.

Discovered tools are mapped to :class:`~agentspec.models.api.APIEndpoint`
objects with ``path="/tools/{tool_name}"`` and ``method=POST``, making them
compatible with the rest of the agentspec pipeline (test generation, running,
reporting).

Usage::

    from agentspec.core.mcp_discovery import discover_mcp

    spec = await discover_mcp("server.py")       # Python source
    spec = await discover_mcp("manifest.json")    # JSON manifest
"""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any

from agentspec.models.api import (
    APIEndpoint,
    APIParameter,
    APISpec,
    FrameworkType,
    HTTPMethod,
    ParamLocation,
    ParamType,
    ResponseSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Map JSON Schema ``type`` values to the agentspec :class:`ParamType` enum.
_JSON_SCHEMA_TYPE_MAP: dict[str, ParamType] = {
    "string": ParamType.STRING,
    "integer": ParamType.INTEGER,
    "number": ParamType.FLOAT,
    "boolean": ParamType.BOOLEAN,
    "array": ParamType.ARRAY,
    "object": ParamType.OBJECT,
}

# Common MCP server object names used in decorator patterns.
_MCP_SERVER_NAMES: frozenset[str] = frozenset({
    "server", "mcp", "app", "mcp_server",
})

# ---------------------------------------------------------------------------
# Helpers -- JSON Schema to APIParameter conversion
# ---------------------------------------------------------------------------


def _resolve_json_schema_type(schema: dict[str, Any]) -> ParamType:
    """Map a JSON Schema ``type`` (and optional ``format``) to :class:`ParamType`.

    Falls back to :attr:`ParamType.ANY` for unknown or missing types.
    """
    type_str = schema.get("type", "")
    fmt = schema.get("format", "")

    if type_str == "string" and fmt == "binary":
        return ParamType.FILE

    return _JSON_SCHEMA_TYPE_MAP.get(type_str, ParamType.ANY)


def _extract_constraints(schema: dict[str, Any]) -> dict[str, Any]:
    """Extract validation constraints from a JSON Schema property definition.

    Returns a dict compatible with :attr:`APIParameter.constraints`.
    """
    constraints: dict[str, Any] = {}

    _MAPPING: dict[str, str] = {
        "minLength": "min_length",
        "maxLength": "max_length",
        "minimum": "minimum",
        "maximum": "maximum",
        "exclusiveMinimum": "exclusive_minimum",
        "exclusiveMaximum": "exclusive_maximum",
        "pattern": "pattern",
        "enum": "enum_values",
        "minItems": "min_items",
        "maxItems": "max_items",
        "default": "default",
    }

    for json_key, constraint_key in _MAPPING.items():
        if json_key in schema:
            constraints[constraint_key] = schema[json_key]

    return constraints


def _parse_schema_properties(
    input_schema: dict[str, Any],
) -> list[APIParameter]:
    """Convert JSON Schema ``properties`` into a list of :class:`APIParameter`.

    Each property becomes a BODY parameter.  The ``required`` list on the
    schema object determines which properties are mandatory.
    """
    properties: dict[str, Any] = input_schema.get("properties", {})
    required_names: set[str] = set(input_schema.get("required", []))
    parameters: list[APIParameter] = []

    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            logger.debug("Skipping non-dict property %r", prop_name)
            continue

        param_type = _resolve_json_schema_type(prop_schema)
        is_required = prop_name in required_names
        default_value = prop_schema.get("default")
        description = prop_schema.get("description", "")
        constraints = _extract_constraints(prop_schema)

        parameters.append(
            APIParameter(
                name=prop_name,
                location=ParamLocation.BODY,
                param_type=param_type,
                required=is_required,
                default=default_value,
                description=description,
                constraints=constraints,
            )
        )

    return parameters


# ---------------------------------------------------------------------------
# Tool-to-endpoint conversion
# ---------------------------------------------------------------------------


def _parse_tool_schema(
    name: str,
    description: str,
    input_schema: dict[str, Any],
) -> APIEndpoint:
    """Convert a single MCP tool definition into an :class:`APIEndpoint`.

    The tool is represented as a virtual ``POST /tools/{name}`` endpoint with
    the input schema mapped to body parameters.

    Parameters
    ----------
    name:
        Tool name (e.g. ``"search"``).
    description:
        Human-readable description of the tool.
    input_schema:
        JSON Schema object describing the tool's input parameters.

    Returns
    -------
    APIEndpoint
        A fully populated endpoint object.
    """
    parameters = _parse_schema_properties(input_schema)
    virtual_path = f"/tools/{name}"

    # Build a default 200 response since MCP tools don't declare response
    # schemas in tool definitions.
    responses = [
        ResponseSchema(
            status_code=200,
            content_type="application/json",
            description="Successful tool invocation",
        ),
    ]

    return APIEndpoint(
        path=virtual_path,
        method=HTTPMethod.POST,
        summary=description,
        description=description,
        parameters=parameters,
        request_body_schema=input_schema if input_schema else None,
        responses=responses,
        tags=["mcp-tool"],
    )


# ---------------------------------------------------------------------------
# AST helpers -- extract tool definitions from Python source
# ---------------------------------------------------------------------------


def _is_mcp_decorator(decorator: ast.expr, handler_name: str) -> bool:
    """Return *True* if *decorator* matches ``@server.<handler_name>()``.

    Recognises patterns such as:
    * ``@server.list_tools()``
    * ``@mcp.call_tool()``
    * ``@app.list_tools()``

    Both ``@server.list_tools()`` (Call) and ``@server.list_tools``
    (bare Attribute) forms are handled.
    """
    # ``@server.list_tools()`` -- Call wrapping an Attribute.
    if isinstance(decorator, ast.Call):
        func = decorator.func
        if isinstance(func, ast.Attribute) and func.attr == handler_name:
            if isinstance(func.value, ast.Name):
                return func.value.id.lower() in _MCP_SERVER_NAMES
        return False

    # ``@server.list_tools`` -- bare Attribute (no parentheses).
    if isinstance(decorator, ast.Attribute) and decorator.attr == handler_name:
        if isinstance(decorator.value, ast.Name):
            return decorator.value.id.lower() in _MCP_SERVER_NAMES
    return False


def _safe_literal_eval(node: ast.expr) -> Any:
    """Attempt to evaluate an AST node as a Python literal.

    Returns ``None`` if the node is not a simple literal (e.g. it contains
    variable references or function calls).
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, RecursionError):
        return None


def _extract_tool_from_call(call_node: ast.Call) -> dict[str, Any] | None:
    """Extract ``name``, ``description``, and ``inputSchema`` from a ``Tool(...)`` call.

    Returns a dict with those keys, or ``None`` if extraction fails.
    """
    # We expect the call to be ``Tool(name=..., description=..., inputSchema=...)``.
    # Both positional and keyword forms are handled.
    name: str | None = None
    description: str = ""
    input_schema: dict[str, Any] = {}

    # Check that this is actually a ``Tool(...)`` call.
    if isinstance(call_node.func, ast.Name) and call_node.func.id == "Tool":
        pass
    elif isinstance(call_node.func, ast.Attribute) and call_node.func.attr == "Tool":
        pass
    else:
        return None

    # Try positional args first (name, description, inputSchema).
    if len(call_node.args) >= 1:
        val = _safe_literal_eval(call_node.args[0])
        if isinstance(val, str):
            name = val
    if len(call_node.args) >= 2:
        val = _safe_literal_eval(call_node.args[1])
        if isinstance(val, str):
            description = val
    if len(call_node.args) >= 3:
        val = _safe_literal_eval(call_node.args[2])
        if isinstance(val, dict):
            input_schema = val

    # Override with keyword arguments.
    for kw in call_node.keywords:
        if kw.arg == "name":
            val = _safe_literal_eval(kw.value)
            if isinstance(val, str):
                name = val
        elif kw.arg == "description":
            val = _safe_literal_eval(kw.value)
            if isinstance(val, str):
                description = val
        elif kw.arg == "inputSchema" or kw.arg == "input_schema":
            val = _safe_literal_eval(kw.value)
            if isinstance(val, dict):
                input_schema = val

    if name is None:
        return None

    return {
        "name": name,
        "description": description,
        "inputSchema": input_schema,
    }


def _extract_tools_from_list_handler(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[dict[str, Any]]:
    """Walk a ``list_tools`` handler function body and extract ``Tool(...)`` definitions.

    Looks for ``Tool(...)`` calls inside:
    * ``return [Tool(...), Tool(...)]``
    * Assignment followed by ``return tools``
    * Any ``Tool(...)`` call nested in the function body.
    """
    tools: list[dict[str, Any]] = []

    for node in ast.walk(func_node):
        if not isinstance(node, ast.Call):
            continue
        tool = _extract_tool_from_call(node)
        if tool is not None:
            tools.append(tool)

    return tools


def _extract_tool_names_from_call_handler(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """Extract tool names dispatched in a ``call_tool`` handler.

    Looks for patterns like::

        if name == "tool_name":
            ...
        elif name == "other_tool":
            ...

    Returns a list of discovered tool name strings.
    """
    names: list[str] = []

    for node in ast.walk(func_node):
        # ``if name == "tool_name"`` → Compare node.
        if isinstance(node, ast.Compare):
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                    # Check that the left side looks like a simple name variable.
                    if isinstance(node.left, ast.Name) and node.left.id in ("name", "tool_name"):
                        names.append(comparator.value)

        # ``match name:`` / ``case "tool_name":`` (Python 3.10+).
        if isinstance(node, ast.Match):
            for case in node.cases:
                if isinstance(case.pattern, ast.MatchValue):
                    if isinstance(case.pattern.value, ast.Constant) and isinstance(
                        case.pattern.value.value, str
                    ):
                        names.append(case.pattern.value.value)

    return names


def _detect_mcp_server_name(source: str) -> str:
    """Try to extract the server name from ``Server("name")`` in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # ``Server("my-server")`` or ``Server(name="my-server")``
        func = node.func
        is_server_call = False
        if isinstance(func, ast.Name) and func.id == "Server":
            is_server_call = True
        elif isinstance(func, ast.Attribute) and func.attr == "Server":
            is_server_call = True

        if not is_server_call:
            continue

        # Try positional arg.
        if node.args:
            val = _safe_literal_eval(node.args[0])
            if isinstance(val, str):
                return val
        # Try keyword.
        for kw in node.keywords:
            if kw.arg == "name":
                val = _safe_literal_eval(kw.value)
                if isinstance(val, str):
                    return val

    return ""


def _parse_mcp_source(source: str, source_path: str) -> list[APIEndpoint]:
    """Parse a Python MCP server source and return discovered tool endpoints.

    Walks the AST looking for:
    1. ``@server.list_tools()`` handlers containing ``Tool(...)`` definitions.
    2. ``@server.call_tool()`` handlers with ``if name == "..."`` dispatching.

    Tools from (1) are fully described (name, description, schema).  Tools
    from (2) only have names -- they are included with empty schemas so that
    the test generator can still create basic invocation tests.

    Parameters
    ----------
    source:
        The Python source code as a string.
    source_path:
        Filesystem path (used only for logging).

    Returns
    -------
    list[APIEndpoint]
        All MCP tool endpoints discovered.
    """
    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        logger.warning("Syntax error parsing %s: %s", source_path, exc)
        return []

    endpoints: list[APIEndpoint] = []
    seen_names: set[str] = set()

    # --- Pass 1: extract fully-described tools from list_tools handlers ---
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for decorator in node.decorator_list:
            if _is_mcp_decorator(decorator, "list_tools"):
                tools = _extract_tools_from_list_handler(node)
                for tool in tools:
                    tool_name = tool["name"]
                    if tool_name in seen_names:
                        continue
                    seen_names.add(tool_name)

                    endpoint = _parse_tool_schema(
                        name=tool_name,
                        description=tool["description"],
                        input_schema=tool["inputSchema"],
                    )
                    endpoints.append(endpoint)
                    logger.debug(
                        "Discovered MCP tool %r from list_tools handler in %s",
                        tool_name,
                        source_path,
                    )

    # --- Pass 2: discover tool names from call_tool dispatchers ---
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for decorator in node.decorator_list:
            if _is_mcp_decorator(decorator, "call_tool"):
                dispatch_names = _extract_tool_names_from_call_handler(node)
                for tool_name in dispatch_names:
                    if tool_name in seen_names:
                        continue
                    seen_names.add(tool_name)

                    # We only have the name -- no description or schema.
                    endpoint = _parse_tool_schema(
                        name=tool_name,
                        description=f"MCP tool: {tool_name}",
                        input_schema={},
                    )
                    endpoints.append(endpoint)
                    logger.debug(
                        "Discovered MCP tool %r from call_tool dispatcher in %s",
                        tool_name,
                        source_path,
                    )

    return endpoints


# ---------------------------------------------------------------------------
# JSON manifest parsing
# ---------------------------------------------------------------------------


def _parse_mcp_manifest(data: dict[str, Any], manifest_path: str) -> list[APIEndpoint]:
    """Parse tool definitions from a loaded MCP manifest dict.

    The expected format is::

        {
          "tools": [
            {
              "name": "search",
              "description": "Search for items",
              "inputSchema": { "type": "object", "properties": {...}, ... }
            },
            ...
          ]
        }

    Parameters
    ----------
    data:
        The parsed JSON content.
    manifest_path:
        Filesystem path (used only for logging).

    Returns
    -------
    list[APIEndpoint]
        Tool endpoints parsed from the manifest.
    """
    tools_raw = data.get("tools", [])
    if not isinstance(tools_raw, list):
        logger.warning(
            "Expected 'tools' to be a list in %s, got %s",
            manifest_path,
            type(tools_raw).__name__,
        )
        return []

    endpoints: list[APIEndpoint] = []

    for idx, tool_def in enumerate(tools_raw):
        if not isinstance(tool_def, dict):
            logger.debug("Skipping non-dict tool entry at index %d in %s", idx, manifest_path)
            continue

        name = tool_def.get("name")
        if not name or not isinstance(name, str):
            logger.debug("Skipping tool with missing/invalid name at index %d in %s", idx, manifest_path)
            continue

        description = tool_def.get("description", "")
        input_schema = tool_def.get("inputSchema", tool_def.get("input_schema", {}))
        if not isinstance(input_schema, dict):
            input_schema = {}

        endpoint = _parse_tool_schema(
            name=name,
            description=description,
            input_schema=input_schema,
        )
        endpoints.append(endpoint)
        logger.debug("Parsed MCP tool %r from manifest %s", name, manifest_path)

    return endpoints


# ---------------------------------------------------------------------------
# File-type detection helpers
# ---------------------------------------------------------------------------


def _source_contains_mcp(source: str) -> bool:
    """Return *True* if the Python source appears to import or use MCP."""
    lower = source.lower()
    return (
        "mcp" in lower
        or "from mcp" in source
        or "import mcp" in source
        or "list_tools" in source
        or "call_tool" in source
    )


def _is_mcp_manifest(data: dict[str, Any]) -> bool:
    """Return *True* if *data* looks like an MCP tool manifest."""
    return "tools" in data and isinstance(data.get("tools"), list)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def discover_mcp_source(source_path: str) -> APISpec:
    """Parse an MCP server Python source file using AST analysis.

    Extracts tool definitions from ``@server.list_tools()`` handlers and
    tool names from ``@server.call_tool()`` dispatchers.

    Parameters
    ----------
    source_path:
        Filesystem path to a ``.py`` file containing an MCP server.

    Returns
    -------
    APISpec
        A specification object with all discovered MCP tool endpoints.

    Raises
    ------
    FileNotFoundError
        If *source_path* does not exist.
    """
    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MCP source file not found: {path}")

    source = path.read_text(encoding="utf-8")
    logger.info("Parsing MCP server source: %s", path)

    endpoints = _parse_mcp_source(source, str(path))
    server_name = _detect_mcp_server_name(source) or path.stem

    return APISpec(
        name=server_name,
        version="0.0.0",
        description=f"MCP server discovered from {path.name}",
        framework=FrameworkType.MCP,
        endpoints=endpoints,
        source_path=str(path),
    )


async def discover_mcp_manifest(manifest_path: str) -> APISpec:
    """Parse MCP tool definitions from a JSON manifest file.

    The manifest should have a top-level ``tools`` array containing objects
    with ``name``, ``description``, and ``inputSchema`` keys.

    Parameters
    ----------
    manifest_path:
        Filesystem path to the JSON manifest.

    Returns
    -------
    APISpec
        A specification object with all parsed MCP tool endpoints.

    Raises
    ------
    FileNotFoundError
        If *manifest_path* does not exist.
    ValueError
        If the file cannot be parsed as JSON or does not contain a ``tools``
        array.
    """
    path = Path(manifest_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"MCP manifest file not found: {path}")

    text = path.read_text(encoding="utf-8")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object at the top level of {path}")

    if not _is_mcp_manifest(data):
        raise ValueError(
            f"{path} does not appear to be an MCP manifest "
            "(missing 'tools' array)"
        )

    logger.info("Parsing MCP manifest: %s", path)
    endpoints = _parse_mcp_manifest(data, str(path))

    # Extract optional metadata from the manifest.
    name = data.get("name", data.get("server_name", path.stem))
    version = data.get("version", "0.0.0")
    description = data.get("description", "")

    return APISpec(
        name=name,
        version=version,
        description=description,
        framework=FrameworkType.MCP,
        endpoints=endpoints,
        source_path=str(path),
    )


async def discover_mcp(source_path: str) -> APISpec:
    """Discover MCP tools from a Python source file or JSON manifest.

    Auto-detects the file type:

    * ``.py`` files are parsed as MCP server source using AST analysis.
    * ``.json`` files are parsed as MCP tool manifests.

    Parameters
    ----------
    source_path:
        Filesystem path to either a Python MCP server file or a JSON
        manifest file.

    Returns
    -------
    APISpec
        A specification with ``framework=FrameworkType.MCP`` and endpoints
        representing each discovered tool.

    Raises
    ------
    FileNotFoundError
        If *source_path* does not exist.
    ValueError
        If the file type is unsupported or the content cannot be parsed.

    Examples
    --------
    ::

        spec = await discover_mcp("my_mcp_server.py")
        print(f"Found {spec.endpoint_count} MCP tools")

        for ep in spec.endpoints:
            print(f"  {ep.path} -- {ep.summary}")
    """
    path = Path(source_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".py":
        source = path.read_text(encoding="utf-8")
        if not _source_contains_mcp(source):
            logger.debug(
                "%s does not appear to contain MCP server code -- skipping",
                path,
            )
            return APISpec(
                name=path.stem,
                framework=FrameworkType.MCP,
                source_path=str(path),
            )
        return await discover_mcp_source(str(path))

    if suffix == ".json":
        return await discover_mcp_manifest(str(path))

    raise ValueError(
        f"Unsupported file type for MCP discovery: {suffix!r}. "
        f"Expected .py (MCP server source) or .json (tool manifest)."
    )
