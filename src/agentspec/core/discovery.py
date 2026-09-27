"""API discovery engine — detect and parse API definitions from source files and specs.

Supports two discovery strategies:

1. **OpenAPI / Swagger parsing** — read ``openapi.json``, ``swagger.yaml``, or
   similar specification files and extract every endpoint.
2. **FastAPI static analysis** — walk Python source with :mod:`ast` to find
   route decorators (``@app.get``, ``@router.post``, ...) and derive
   parameters, request bodies, and response models from type hints.

The top-level :func:`discover` function auto-detects which strategy to use.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from pathlib import Path
from typing import Any, Sequence

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
# Public constants
# ---------------------------------------------------------------------------

_HTTP_METHODS: frozenset[str] = frozenset(m.value.lower() for m in HTTPMethod)

_OPENAPI_EXTENSIONS: frozenset[str] = frozenset({".json", ".yaml", ".yml"})

_PYTHON_TYPE_MAP: dict[str, ParamType] = {
    "str": ParamType.STRING,
    "int": ParamType.INTEGER,
    "float": ParamType.FLOAT,
    "bool": ParamType.BOOLEAN,
    "list": ParamType.ARRAY,
    "List": ParamType.ARRAY,
    "dict": ParamType.OBJECT,
    "Dict": ParamType.OBJECT,
    "set": ParamType.ARRAY,
    "Set": ParamType.ARRAY,
    "tuple": ParamType.ARRAY,
    "Tuple": ParamType.ARRAY,
    "bytes": ParamType.STRING,
    "UploadFile": ParamType.FILE,
    "Any": ParamType.ANY,
}

_OPENAPI_TYPE_MAP: dict[str, ParamType] = {
    "string": ParamType.STRING,
    "integer": ParamType.INTEGER,
    "number": ParamType.FLOAT,
    "boolean": ParamType.BOOLEAN,
    "array": ParamType.ARRAY,
    "object": ParamType.OBJECT,
    "file": ParamType.FILE,
}

_OPENAPI_LOCATION_MAP: dict[str, ParamLocation] = {
    "path": ParamLocation.PATH,
    "query": ParamLocation.QUERY,
    "header": ParamLocation.HEADER,
    "cookie": ParamLocation.COOKIE,
}

# Matches ``{param_name}`` inside a URL path.
_PATH_PARAM_RE: re.Pattern[str] = re.compile(r"\{(\w+)\}")

# Route-bearing instance names commonly used in FastAPI projects.
_FASTAPI_APP_NAMES: frozenset[str] = frozenset({
    "app", "router", "api", "api_router", "v1_router", "v1", "v2",
})

# ---------------------------------------------------------------------------
# YAML loader helper
# ---------------------------------------------------------------------------


def _load_yaml(text: str) -> Any:
    """Load YAML text, falling back to JSON if *PyYAML* is unavailable."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("PyYAML not installed — attempting JSON parse as fallback")
        return json.loads(text)
    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Helpers — type mapping
# ---------------------------------------------------------------------------


def _resolve_python_type(annotation: ast.expr | None) -> ParamType:
    """Map a Python AST type annotation to a :class:`ParamType`.

    Handles bare names (``int``), subscripts (``list[int]``), ``Optional[X]``,
    and ``Union[X, None]`` — returning the *inner* type for optionals.
    """
    if annotation is None:
        return ParamType.ANY

    # ``int``, ``str``, ``UploadFile``, ...
    if isinstance(annotation, ast.Name):
        return _PYTHON_TYPE_MAP.get(annotation.id, ParamType.OBJECT)

    # ``Optional[int]`` appears as ``Subscript(Name('Optional'), ...)``.
    if isinstance(annotation, ast.Subscript):
        if isinstance(annotation.value, ast.Name):
            outer = annotation.value.id
            if outer == "Optional":
                return _resolve_python_type(annotation.slice)
            if outer in ("List", "list", "Set", "set", "Sequence", "Tuple", "tuple"):
                return ParamType.ARRAY
            if outer in ("Dict", "dict", "Mapping"):
                return ParamType.OBJECT
        # ``Union[X, None]`` — pick the first non-None element.
        if isinstance(annotation.value, ast.Name) and annotation.value.id == "Union":
            if isinstance(annotation.slice, ast.Tuple):
                for elt in annotation.slice.elts:
                    if isinstance(elt, ast.Constant) and elt.value is None:
                        continue
                    if isinstance(elt, ast.Name) and elt.id == "None":
                        continue
                    return _resolve_python_type(elt)
        return ParamType.OBJECT

    # ``X | None`` (Python 3.10+) → ast.BinOp with BitOr
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        left_type = _resolve_python_type(annotation.left)
        if left_type is not ParamType.ANY:
            return left_type
        return _resolve_python_type(annotation.right)

    # ``ast.Attribute`` — e.g. ``pydantic.BaseModel``.  Treat as OBJECT.
    if isinstance(annotation, ast.Attribute):
        return ParamType.OBJECT

    return ParamType.ANY


def _resolve_openapi_type(schema: dict[str, Any]) -> ParamType:
    """Map an OpenAPI schema ``type`` (and optional ``format``) to :class:`ParamType`."""
    type_str = schema.get("type", "")
    fmt = schema.get("format", "")
    if type_str == "string" and fmt == "binary":
        return ParamType.FILE
    if type_str == "number":
        return ParamType.FLOAT
    return _OPENAPI_TYPE_MAP.get(type_str, ParamType.ANY)


# ---------------------------------------------------------------------------
# Helpers — AST utilities
# ---------------------------------------------------------------------------


def _get_decorator_method_and_path(
    decorator: ast.expr,
) -> tuple[str, str] | None:
    """Extract ``(http_method, path)`` from a FastAPI-style decorator.

    Recognises patterns:
    * ``@app.get("/path")``
    * ``@router.post("/path")``
    * ``@api.delete("/path", ...)``

    Returns ``None`` if the decorator does not match.
    """
    if not isinstance(decorator, ast.Call):
        return None
    func = decorator.func
    if not isinstance(func, ast.Attribute):
        return None
    method_name = func.attr.lower()
    if method_name not in _HTTP_METHODS:
        return None

    # Accept known instance names, but also any attribute call with a valid
    # HTTP method name — covers ``some_router.get(...)`` and similar.
    if isinstance(func.value, ast.Name):
        instance_name = func.value.id.lower()
        # Be permissive: if the method is a valid HTTP verb we accept it
        # regardless of the instance variable name, but log unknown ones.
        if instance_name not in _FASTAPI_APP_NAMES:
            logger.debug(
                "Accepting route decorator on unfamiliar object %r",
                func.value.id,
            )
    else:
        return None

    # The first positional arg is the path string.
    if not decorator.args:
        return None
    path_node = decorator.args[0]
    if isinstance(path_node, ast.Constant) and isinstance(path_node.value, str):
        return method_name.upper(), path_node.value

    return None


def _extract_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, str]:
    """Return ``(summary, description)`` extracted from a function docstring.

    The first line becomes the *summary*; everything else becomes the
    *description*.
    """
    raw = ast.get_docstring(node)
    if not raw:
        return "", ""
    lines = raw.strip().splitlines()
    summary = lines[0].strip()
    description = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return summary, description


def _extract_default_value(node: ast.expr) -> Any:
    """Try to recover a literal default value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _is_fastapi_marker(node: ast.expr, marker_name: str) -> bool:
    """Return *True* if *node* is a call to a FastAPI dependency like ``Query(...)``."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == marker_name:
            return True
        if isinstance(node.func, ast.Attribute) and node.func.attr == marker_name:
            return True
    return False


def _param_location_from_default(
    default: ast.expr | None,
    param_name: str,
    path: str,
) -> ParamLocation:
    """Determine parameter location from its default value or position in the path."""
    if default is not None:
        if _is_fastapi_marker(default, "Path"):
            return ParamLocation.PATH
        if _is_fastapi_marker(default, "Query"):
            return ParamLocation.QUERY
        if _is_fastapi_marker(default, "Body"):
            return ParamLocation.BODY
        if _is_fastapi_marker(default, "Header"):
            return ParamLocation.HEADER
        if _is_fastapi_marker(default, "Cookie"):
            return ParamLocation.COOKIE

    # Implicit: if the name appears as ``{name}`` in the path it is a PATH param.
    path_param_names = set(_PATH_PARAM_RE.findall(path))
    if param_name in path_param_names:
        return ParamLocation.PATH

    return ParamLocation.QUERY


def _is_required_param(default: ast.expr | None) -> bool:
    """Determine whether a function argument is required.

    An argument is required if it has no default *or* its default is the
    ``...`` sentinel (``Ellipsis``), which FastAPI uses for required
    dependency-injected params like ``Query(...)``.
    """
    if default is None:
        return True
    if isinstance(default, ast.Constant) and default.value is ...:
        return True
    # ``Query(...)`` — check first positional arg.
    if isinstance(default, ast.Call) and default.args:
        first = default.args[0]
        if isinstance(first, ast.Constant) and first.value is ...:
            return True
    return False


def _extract_marker_description(node: ast.expr) -> str:
    """Pull out the ``description=`` keyword from a FastAPI marker call."""
    if not isinstance(node, ast.Call):
        return ""
    for kw in node.keywords:
        if kw.arg == "description" and isinstance(kw.value, ast.Constant):
            return str(kw.value.value)
    return ""


def _extract_marker_constraints(node: ast.expr) -> dict[str, Any]:
    """Extract constraint keywords (min_length, ge, le, ...) from a marker call."""
    constraints: dict[str, Any] = {}
    if not isinstance(node, ast.Call):
        return constraints
    _KNOWN = {
        "min_length", "max_length", "gt", "ge", "lt", "le",
        "regex", "pattern", "multiple_of", "example",
    }
    for kw in node.keywords:
        if kw.arg in _KNOWN and isinstance(kw.value, ast.Constant):
            constraints[kw.arg] = kw.value.value
    return constraints


def _marker_default_value(node: ast.expr) -> Any:
    """Extract the *default* value from a FastAPI marker like ``Query(default=5)``."""
    if not isinstance(node, ast.Call):
        return None
    # Positional: Query("default") — but skip ``...``
    if node.args:
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value is not ...:
            return first.value
    # Keyword: Query(default=5)
    for kw in node.keywords:
        if kw.arg == "default" and isinstance(kw.value, ast.Constant):
            return kw.value.value
    return None


# ---------------------------------------------------------------------------
# Helpers — type hint classification for body detection
# ---------------------------------------------------------------------------

# Simple / scalar type names that should NOT be treated as a request body.
_SCALAR_TYPE_NAMES: frozenset[str] = frozenset({
    "str", "int", "float", "bool", "bytes",
    "date", "datetime", "time", "timedelta",
    "UUID", "Decimal", "Path",
})


def _looks_like_body_type(annotation: ast.expr | None) -> bool:
    """Heuristic: return *True* if the annotation looks like a Pydantic model.

    We cannot resolve imports during static analysis, so we assume any
    non-scalar, non-generic Name that is **not** a known FastAPI special type
    is a Pydantic model intended as the request body.
    """
    if annotation is None:
        return False
    if isinstance(annotation, ast.Name):
        return annotation.id not in _SCALAR_TYPE_NAMES and annotation.id not in _PYTHON_TYPE_MAP
    if isinstance(annotation, ast.Attribute):
        return True  # e.g. ``schemas.ItemCreate``
    return False


# Types that serialise as strings in JSON but are not in _PYTHON_TYPE_MAP.
_STRING_LIKE_TYPES: frozenset[str] = frozenset({
    "date", "datetime", "time", "timedelta", "UUID", "Decimal",
})


def _unwrap_optional(annotation: ast.expr) -> ast.expr:
    """Strip ``Optional[X]`` or ``X | None`` wrapping, returning the inner type."""
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        if annotation.value.id == "Optional":
            return annotation.slice
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        right = annotation.right
        left = annotation.left
        if (isinstance(right, ast.Constant) and right.value is None) or (
            isinstance(right, ast.Name) and right.id == "None"
        ):
            return left
        if (isinstance(left, ast.Constant) and left.value is None) or (
            isinstance(left, ast.Name) and left.id == "None"
        ):
            return right
    return annotation


def _is_optional_annotation(annotation: ast.expr) -> bool:
    """Return *True* if the annotation is ``Optional[X]`` or ``X | None``."""
    if isinstance(annotation, ast.Subscript) and isinstance(annotation.value, ast.Name):
        if annotation.value.id == "Optional":
            return True
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        right = annotation.right
        left = annotation.left
        if (isinstance(right, ast.Constant) and right.value is None) or (
            isinstance(right, ast.Name) and right.id == "None"
        ):
            return True
        if (isinstance(left, ast.Constant) and left.value is None) or (
            isinstance(left, ast.Name) and left.id == "None"
        ):
            return True
    return False


def _extract_pydantic_models(
    tree: ast.Module,
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    """Extract Pydantic model schemas and enum value maps from the AST.

    Returns ``(models, enum_values)`` where *enum_values* maps enum class
    names to their string member values.

    Discovers :class:`~enum.Enum` subclasses first so that enum-typed fields
    can include their allowed values.  Then finds :class:`~pydantic.BaseModel`
    subclasses and builds a JSON-Schema-like dict for each.

    Returns a mapping of *class name* -> schema dict.
    """
    # --- Pass 1: discover enum classes ---
    enum_values: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_enum = any(
            (isinstance(b, ast.Name) and b.id == "Enum")
            or (isinstance(b, ast.Attribute) and b.attr == "Enum")
            for b in node.bases
        )
        if not is_enum:
            continue
        values: list[str] = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                if isinstance(item.value, ast.Constant) and isinstance(
                    item.value.value, str
                ):
                    values.append(item.value.value)
        if values:
            enum_values[node.name] = values

    # --- Pass 2: discover BaseModel subclasses ---
    models: dict[str, dict[str, Any]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_model = any(
            (isinstance(b, ast.Name) and b.id == "BaseModel")
            or (isinstance(b, ast.Attribute) and b.attr == "BaseModel")
            for b in node.bases
        )
        if not is_model:
            continue

        properties: dict[str, Any] = {}
        required_fields: list[str] = []

        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(
                item.target, ast.Name
            ):
                continue

            field_name = item.target.id
            annotation = item.annotation

            # Resolve core type.
            field_type = _resolve_python_type(annotation)

            # Unwrap Optional / X | None to inspect the inner type.
            inner = _unwrap_optional(annotation)

            # Check if annotation references a known enum.
            enum_vals: list[str] | None = None
            if isinstance(inner, ast.Name) and inner.id in enum_values:
                enum_vals = enum_values[inner.id]
                field_type = ParamType.STRING  # enums serialise as strings
            elif isinstance(inner, ast.Name) and inner.id in _STRING_LIKE_TYPES:
                # Handle date-like types that serialise as strings.
                field_type = ParamType.STRING

            # Determine whether the field is required.
            is_required: bool = item.value is None  # no default -> required
            if item.value is not None:
                if isinstance(item.value, ast.Call):
                    # Field(...) -> required; Field(None, ...) -> optional
                    if item.value.args:
                        first = item.value.args[0]
                        if isinstance(first, ast.Constant) and first.value is ...:
                            is_required = True
                        else:
                            is_required = False
                    else:
                        is_required = False
                else:
                    is_required = False

            # Optional type annotation always means not required.
            if _is_optional_annotation(annotation):
                is_required = False

            prop: dict[str, Any] = {"type": field_type.value}
            if enum_vals:
                prop["enum"] = enum_vals
            properties[field_name] = prop

            if is_required:
                required_fields.append(field_name)

        if properties:
            models[node.name] = {
                "type": "object",
                "properties": properties,
                "required": required_fields,
            }

    return models, enum_values


# ---------------------------------------------------------------------------
# FastAPI AST discovery
# ---------------------------------------------------------------------------


def _parse_function_params(
    func: ast.FunctionDef | ast.AsyncFunctionDef,
    route_path: str,
    pydantic_models: dict[str, dict[str, Any]] | None = None,
    enum_values: dict[str, list[str]] | None = None,
) -> tuple[list[APIParameter], dict[str, Any] | None]:
    """Parse function arguments into API parameters and an optional request body schema.

    Returns ``(parameters, request_body_schema)``.
    """
    parameters: list[APIParameter] = []
    request_body_schema: dict[str, Any] | None = None

    args = func.args

    # Build a mapping of argument-name → default-node.  ``ast.arguments``
    # stores defaults only for the *last N* positional args, so we need to
    # align them carefully.
    num_args = len(args.args)
    num_defaults = len(args.defaults)
    defaults_offset = num_args - num_defaults

    def _default_for(idx: int) -> ast.expr | None:
        di = idx - defaults_offset
        if di >= 0:
            return args.defaults[di]
        return None

    path_param_names = set(_PATH_PARAM_RE.findall(route_path))
    body_fields: dict[str, Any] = {}

    for idx, arg in enumerate(args.args):
        name = arg.arg

        # Skip ``self`` / ``cls`` and common FastAPI injected dependencies.
        if name in ("self", "cls", "request", "response", "db", "session", "background_tasks"):
            continue

        annotation = arg.annotation
        default_node = _default_for(idx)

        # Determine location -----------------------------------------------
        location = _param_location_from_default(default_node, name, route_path)

        # If annotated with a model-like type AND not explicitly marked, treat
        # as body.
        if (
            location == ParamLocation.QUERY
            and name not in path_param_names
            and _looks_like_body_type(annotation)
            and (default_node is None or not isinstance(default_node, ast.Call))
        ):
            location = ParamLocation.BODY

        # Resolve type hint -------------------------------------------------
        param_type = _resolve_python_type(annotation)

        # Required? ---------------------------------------------------------
        required = _is_required_param(default_node)

        # Description & constraints from markers ----------------------------
        description = ""
        constraints: dict[str, Any] = {}
        default_value: Any = None

        if default_node is not None and isinstance(default_node, ast.Call):
            description = _extract_marker_description(default_node)
            constraints = _extract_marker_constraints(default_node)
            default_value = _marker_default_value(default_node)
        elif default_node is not None:
            default_value = _extract_default_value(default_node)

        if location == ParamLocation.BODY:
            # If the annotation references a known Pydantic model, use its
            # fully-resolved schema instead of a single opaque field.
            _model_name: str | None = None
            if isinstance(annotation, ast.Name):
                _model_name = annotation.id
            elif isinstance(annotation, ast.Attribute):
                _model_name = annotation.attr
            if _model_name and pydantic_models and _model_name in pydantic_models:
                request_body_schema = pydantic_models[_model_name]
                continue

            # Accumulate fields for a body schema instead of individual params.
            body_fields[name] = {
                "type": param_type.value,
                "required": required,
            }
            if description:
                body_fields[name]["description"] = description
            if default_value is not None:
                body_fields[name]["default"] = default_value
            continue

        # If the annotation references a known Enum type, inject its values
        # into constraints so the test generator uses a valid value.
        if enum_values and annotation is not None and "enum_values" not in constraints:
            inner = _unwrap_optional(annotation)
            _enum_name: str | None = None
            if isinstance(inner, ast.Name):
                _enum_name = inner.id
            elif isinstance(inner, ast.Attribute):
                _enum_name = inner.attr
            if _enum_name and _enum_name in enum_values:
                constraints["enum_values"] = enum_values[_enum_name]

        parameters.append(
            APIParameter(
                name=name,
                location=location,
                param_type=param_type,
                required=required,
                default=default_value,
                description=description,
                constraints=constraints,
            )
        )

    # keyword-only args (after ``*`` in the signature) are usually Query params.
    kw_defaults = args.kw_defaults  # same length as kwonlyargs, may contain None
    for idx, arg in enumerate(args.kwonlyargs):
        name = arg.arg
        if name in ("self", "cls", "request", "response", "db", "session", "background_tasks"):
            continue

        annotation = arg.annotation
        default_node = kw_defaults[idx] if idx < len(kw_defaults) else None

        location = _param_location_from_default(default_node, name, route_path)
        param_type = _resolve_python_type(annotation)
        required = _is_required_param(default_node)

        description = ""
        constraints: dict[str, Any] = {}
        default_value = None

        if default_node is not None and isinstance(default_node, ast.Call):
            description = _extract_marker_description(default_node)
            constraints = _extract_marker_constraints(default_node)
            default_value = _marker_default_value(default_node)
        elif default_node is not None:
            default_value = _extract_default_value(default_node)

        # Enum resolution for kwonly args (same logic as positional args).
        if enum_values and annotation is not None and "enum_values" not in constraints:
            inner = _unwrap_optional(annotation)
            _kw_enum_name: str | None = None
            if isinstance(inner, ast.Name):
                _kw_enum_name = inner.id
            elif isinstance(inner, ast.Attribute):
                _kw_enum_name = inner.attr
            if _kw_enum_name and _kw_enum_name in enum_values:
                constraints["enum_values"] = enum_values[_kw_enum_name]

        parameters.append(
            APIParameter(
                name=name,
                location=location,
                param_type=param_type,
                required=required,
                default=default_value,
                description=description,
                constraints=constraints,
            )
        )

    if body_fields and request_body_schema is None:
        request_body_schema = {
            "type": "object",
            "properties": body_fields,
            "required": [k for k, v in body_fields.items() if v.get("required", True)],
        }

    return parameters, request_body_schema


def _parse_fastapi_source(source: str, source_path: str) -> list[APIEndpoint]:
    """Parse a Python source string and return all FastAPI endpoints found."""
    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        logger.warning("Syntax error parsing %s: %s", source_path, exc)
        return []

    endpoints: list[APIEndpoint] = []
    pydantic_models, enum_values = _extract_pydantic_models(tree)

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        for decorator in node.decorator_list:
            result = _get_decorator_method_and_path(decorator)
            if result is None:
                continue

            method_str, route_path = result

            try:
                method = HTTPMethod(method_str)
            except ValueError:
                logger.debug("Skipping unrecognised HTTP method %r", method_str)
                continue

            summary, description = _extract_docstring(node)
            parameters, request_body_schema = _parse_function_params(
                node, route_path,
                pydantic_models=pydantic_models,
                enum_values=enum_values,
            )

            # Extract tags and other metadata from decorator kwargs.
            tags: list[str] = []
            deprecated = False
            if isinstance(decorator, ast.Call):
                for kw in decorator.keywords:
                    if kw.arg == "tags" and isinstance(kw.value, ast.List):
                        for elt in kw.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                tags.append(elt.value)
                    if kw.arg == "deprecated" and isinstance(kw.value, ast.Constant):
                        deprecated = bool(kw.value.value)

            # Extract response_model hint — store as a basic schema reference.
            responses: list[ResponseSchema] = []
            if isinstance(decorator, ast.Call):
                for kw in decorator.keywords:
                    if kw.arg == "response_model":
                        model_name = ""
                        if isinstance(kw.value, ast.Name):
                            model_name = kw.value.id
                        elif isinstance(kw.value, ast.Attribute):
                            model_name = ast.dump(kw.value)
                        if model_name:
                            responses.append(
                                ResponseSchema(
                                    status_code=200,
                                    content_type="application/json",
                                    schema_def={"$ref": model_name},
                                    description="Successful response",
                                )
                            )

            # Default 200 response when none could be inferred.
            if not responses:
                responses.append(
                    ResponseSchema(
                        status_code=200,
                        content_type="application/json",
                        description="Successful response",
                    )
                )

            endpoints.append(
                APIEndpoint(
                    path=route_path,
                    method=method,
                    summary=summary,
                    description=description,
                    parameters=parameters,
                    request_body_schema=request_body_schema,
                    responses=responses,
                    tags=tags,
                    deprecated=deprecated,
                )
            )

    return endpoints


def _detect_fastapi_app_name(source: str) -> str:
    """Try to extract the API title from ``FastAPI(title=...)`` in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "FastAPI":
            for kw in node.keywords:
                if kw.arg == "title" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
    return ""


def _detect_fastapi_version(source: str) -> str:
    """Try to extract the API version from ``FastAPI(version=...)`` in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "FastAPI":
            for kw in node.keywords:
                if kw.arg == "version" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
    return ""


def _detect_fastapi_description(source: str) -> str:
    """Try to extract the API description from ``FastAPI(description=...)``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == "FastAPI":
            for kw in node.keywords:
                if kw.arg == "description" and isinstance(kw.value, ast.Constant):
                    return str(kw.value.value)
    return ""


# ---------------------------------------------------------------------------
# OpenAPI / Swagger discovery
# ---------------------------------------------------------------------------


def _parse_openapi_parameter(raw: dict[str, Any]) -> APIParameter:
    """Convert a single OpenAPI parameter object to an :class:`APIParameter`."""
    schema = raw.get("schema", {})
    location_str = raw.get("in", "query")
    location = _OPENAPI_LOCATION_MAP.get(location_str, ParamLocation.QUERY)

    param_type = _resolve_openapi_type(schema)

    constraints: dict[str, Any] = {}
    for key in ("minLength", "maxLength", "minimum", "maximum", "pattern", "enum"):
        val = schema.get(key)
        if val is not None:
            # Normalise to snake_case keys used by our model.
            normalised = {
                "minLength": "min_length",
                "maxLength": "max_length",
                "minimum": "minimum",
                "maximum": "maximum",
                "pattern": "pattern",
                "enum": "enum_values",
            }.get(key, key)
            constraints[normalised] = val

    return APIParameter(
        name=raw.get("name", ""),
        location=location,
        param_type=param_type,
        required=raw.get("required", location == ParamLocation.PATH),
        default=schema.get("default"),
        description=raw.get("description", ""),
        constraints=constraints,
    )


def _parse_openapi_request_body(body: dict[str, Any]) -> dict[str, Any] | None:
    """Extract a schema dict from an OpenAPI ``requestBody`` object."""
    content = body.get("content", {})
    for media_type in ("application/json", "multipart/form-data", "application/x-www-form-urlencoded"):
        if media_type in content:
            return content[media_type].get("schema")
    # Fallback — take the first one available.
    for _mt, detail in content.items():
        return detail.get("schema")
    return None


def _parse_openapi_responses(raw: dict[str, Any]) -> list[ResponseSchema]:
    """Convert an OpenAPI ``responses`` mapping to a list of :class:`ResponseSchema`."""
    responses: list[ResponseSchema] = []
    for status_str, detail in raw.items():
        try:
            status_code = int(status_str)
        except (ValueError, TypeError):
            # "default" or other non-numeric keys.
            status_code = 0

        content = detail.get("content", {})
        content_type = "application/json"
        schema_def: dict[str, Any] = {}
        if content:
            # Prefer JSON; take whatever is first otherwise.
            if "application/json" in content:
                content_type = "application/json"
                schema_def = content["application/json"].get("schema", {})
            else:
                ct, ct_detail = next(iter(content.items()))
                content_type = ct
                schema_def = ct_detail.get("schema", {})

        responses.append(
            ResponseSchema(
                status_code=status_code,
                content_type=content_type,
                schema_def=schema_def,
                description=detail.get("description", ""),
            )
        )

    return responses


def _parse_openapi_spec(data: dict[str, Any], source_path: str) -> APISpec:
    """Build an :class:`APISpec` from an already-loaded OpenAPI/Swagger dict."""
    info = data.get("info", {})
    openapi_version = data.get("openapi", data.get("swagger", ""))

    # Determine base_url from ``servers`` (OpenAPI 3.x) or ``host``/``basePath`` (2.x).
    base_url = ""
    servers = data.get("servers")
    if servers and isinstance(servers, list):
        base_url = servers[0].get("url", "")
    else:
        host = data.get("host", "")
        base_path = data.get("basePath", "")
        schemes = data.get("schemes", ["https"])
        if host:
            scheme = schemes[0] if schemes else "https"
            base_url = f"{scheme}://{host}{base_path}"

    # Auth schemes.
    auth_schemes: list[str] = []
    security_defs = (
        data.get("components", {}).get("securitySchemes")
        or data.get("securityDefinitions")
        or {}
    )
    for scheme_name, scheme_def in security_defs.items():
        scheme_type = scheme_def.get("type", "")
        auth_schemes.append(f"{scheme_name}:{scheme_type}")

    # Walk paths.
    endpoints: list[APIEndpoint] = []
    paths = data.get("paths", {})
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue

        # Path-level parameters shared by all operations.
        shared_params_raw: list[dict[str, Any]] = path_item.get("parameters", [])

        for method_str in _HTTP_METHODS:
            operation = path_item.get(method_str)
            if operation is None:
                continue

            try:
                method = HTTPMethod(method_str.upper())
            except ValueError:
                continue

            # Merge path-level + operation-level parameters (operation wins).
            op_params_raw: list[dict[str, Any]] = operation.get("parameters", [])
            seen: set[tuple[str, str]] = set()
            merged_raw: list[dict[str, Any]] = []
            for p in op_params_raw:
                key = (p.get("name", ""), p.get("in", ""))
                seen.add(key)
                merged_raw.append(p)
            for p in shared_params_raw:
                key = (p.get("name", ""), p.get("in", ""))
                if key not in seen:
                    merged_raw.append(p)

            parameters = [_parse_openapi_parameter(p) for p in merged_raw]

            # Request body (OpenAPI 3.x).
            request_body_schema: dict[str, Any] | None = None
            if "requestBody" in operation:
                request_body_schema = _parse_openapi_request_body(operation["requestBody"])

            # Swagger 2.x body parameters.
            if request_body_schema is None:
                for p in merged_raw:
                    if p.get("in") == "body":
                        request_body_schema = p.get("schema")
                        break

            responses = _parse_openapi_responses(operation.get("responses", {}))

            tags = operation.get("tags", [])
            requires_auth = bool(operation.get("security") or data.get("security"))
            deprecated = operation.get("deprecated", False)

            endpoints.append(
                APIEndpoint(
                    path=path,
                    method=method,
                    summary=operation.get("summary", ""),
                    description=operation.get("description", ""),
                    parameters=parameters,
                    request_body_schema=request_body_schema,
                    responses=responses,
                    tags=tags,
                    requires_auth=requires_auth,
                    deprecated=deprecated,
                )
            )

    return APISpec(
        name=info.get("title", "Untitled API"),
        version=info.get("version", "0.0.0"),
        description=info.get("description", ""),
        base_url=base_url,
        framework=FrameworkType.OPENAPI,
        endpoints=endpoints,
        source_path=source_path,
        openapi_version=openapi_version,
        auth_schemes=auth_schemes,
    )


# ---------------------------------------------------------------------------
# File-type detection
# ---------------------------------------------------------------------------


def _is_openapi_content(data: dict[str, Any]) -> bool:
    """Return *True* if *data* looks like an OpenAPI or Swagger spec."""
    return "openapi" in data or "swagger" in data


def _file_looks_like_openapi(path: Path) -> bool:
    """Quick heuristic: does the filename suggest an OpenAPI spec?"""
    name = path.name.lower()
    return any(
        keyword in name
        for keyword in ("openapi", "swagger", "api-spec", "api_spec", "apispec")
    )


def _source_contains_fastapi(source: str) -> bool:
    """Return *True* if the Python source appears to import or use FastAPI."""
    return "fastapi" in source.lower() or "FastAPI" in source


def _source_contains_flask(source: str) -> bool:
    """Return *True* if the Python source appears to use Flask."""
    return (
        "from flask" in source
        or "import flask" in source
        or "Flask(__name__)" in source
        or "@app.route" in source
        or "Blueprint(" in source
    )


def _source_contains_mcp(source: str) -> bool:
    """Return *True* if the Python source appears to define an MCP server."""
    return (
        "mcp.server" in source
        or "from mcp" in source
        or "@server.list_tools" in source
        or "@server.call_tool" in source
    )


_JS_EXTENSIONS: frozenset[str] = frozenset({".js", ".ts", ".mjs", ".cjs", ".tsx", ".jsx"})


def _source_contains_js_routes(source: str) -> bool:
    """Return *True* if a JS/TS source appears to define HTTP routes."""
    lower = source.lower()
    return (
        "require('express')" in lower
        or 'require("express")' in lower
        or "from 'express'" in lower
        or 'from "express"' in lower
        or "@nestjs/common" in lower
        or "from 'hono'" in lower
        or 'from "hono"' in lower
        or "fastify" in lower
        or ".route(" in source
        or re.search(r'(?:app|router|server)\s*\.\s*(?:get|post|put|patch|delete)\s*\(', source) is not None
        or re.search(r'@(?:Get|Post|Put|Patch|Delete)\s*\(', source) is not None
        or re.search(r'export\s+(?:async\s+)?function\s+(?:GET|POST|PUT|PATCH|DELETE)\s*\(', source) is not None
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def discover_openapi(spec_path: str) -> APISpec:
    """Parse an OpenAPI or Swagger specification file.

    Accepts JSON (``.json``) and YAML (``.yaml`` / ``.yml``) files.

    Parameters
    ----------
    spec_path:
        Filesystem path to the specification file.

    Returns
    -------
    APISpec
        A fully populated specification object.

    Raises
    ------
    FileNotFoundError
        If *spec_path* does not exist.
    ValueError
        If the file cannot be parsed or is not a valid OpenAPI document.
    """
    path = Path(spec_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Spec file not found: {path}")

    text = path.read_text(encoding="utf-8")

    if path.suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    else:
        try:
            data = _load_yaml(text)
        except Exception as exc:
            raise ValueError(f"Failed to parse YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the top level of {path}")

    if not _is_openapi_content(data):
        raise ValueError(
            f"{path} does not appear to be an OpenAPI/Swagger spec "
            "(missing 'openapi' or 'swagger' key)"
        )

    logger.info("Parsing OpenAPI spec: %s", path)
    return _parse_openapi_spec(data, str(path))


async def discover_fastapi(source_path: str) -> APISpec:
    """Parse a FastAPI Python source file using AST analysis.

    Parameters
    ----------
    source_path:
        Filesystem path to a ``.py`` file containing FastAPI routes.

    Returns
    -------
    APISpec
        A specification object with all discovered endpoints.

    Raises
    ------
    FileNotFoundError
        If *source_path* does not exist.
    """
    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    source = path.read_text(encoding="utf-8")
    logger.info("Parsing FastAPI source: %s", path)

    endpoints = _parse_fastapi_source(source, str(path))
    name = _detect_fastapi_app_name(source) or path.stem
    version = _detect_fastapi_version(source) or "0.0.0"
    description = _detect_fastapi_description(source)

    return APISpec(
        name=name,
        version=version,
        description=description,
        framework=FrameworkType.FASTAPI,
        endpoints=endpoints,
        source_path=str(path),
    )


async def discover_directory(dir_path: str) -> APISpec:
    """Recursively scan a directory for API definitions and merge results.

    Walks *dir_path* looking for Python files and OpenAPI spec files.  Each
    parseable file contributes its endpoints to the merged :class:`APISpec`.
    Unparseable files are skipped with a warning.

    Parameters
    ----------
    dir_path:
        Filesystem path to the directory to scan.

    Returns
    -------
    APISpec
        A merged specification containing endpoints from all discovered files.

    Raises
    ------
    FileNotFoundError
        If *dir_path* does not exist.
    NotADirectoryError
        If *dir_path* is not a directory.
    """
    root = Path(dir_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    logger.info("Scanning directory: %s", root)

    merged = APISpec(
        name=root.name,
        source_path=str(root),
    )

    _SCAN_EXTENSIONS = {".py", ".json", ".yaml", ".yml"} | _JS_EXTENSIONS

    # Skip dependency / build directories that never contain hand-written routes.
    _SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist", "build", ".next", ".nuxt"}

    # Collect candidate files.
    candidates: list[Path] = sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix in _SCAN_EXTENSIONS
        and not any(part in _SKIP_DIRS for part in p.parts)
    )

    for candidate in candidates:
        # Skip common non-API files.
        if candidate.name.startswith(".") or candidate.name.startswith("__"):
            continue
        if any(
            part.startswith(".")
            or part in ("node_modules", ".venv", "venv", "__pycache__", ".git")
            for part in candidate.parts
        ):
            continue

        try:
            spec = await discover(str(candidate))
            if spec.endpoints:
                merged.endpoints.extend(spec.endpoints)
                # Inherit the framework from the first file that yields endpoints.
                if merged.framework == FrameworkType.UNKNOWN:
                    merged.framework = spec.framework
                # Merge auth schemes.
                for scheme in spec.auth_schemes:
                    if scheme not in merged.auth_schemes:
                        merged.auth_schemes.append(scheme)
                # Prefer a more descriptive name / version if available.
                if spec.name and spec.name != "Untitled API" and merged.name == root.name:
                    merged.name = spec.name
                if spec.version and spec.version != "0.0.0" and merged.version == "0.0.0":
                    merged.version = spec.version
                if spec.description and not merged.description:
                    merged.description = spec.description
                if spec.base_url and not merged.base_url:
                    merged.base_url = spec.base_url
                if spec.openapi_version and not merged.openapi_version:
                    merged.openapi_version = spec.openapi_version

                logger.info(
                    "Discovered %d endpoint(s) in %s",
                    len(spec.endpoints),
                    candidate,
                )
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skipping %s: %s", candidate, exc)

    logger.info(
        "Directory scan complete: %d endpoint(s) from %s",
        len(merged.endpoints),
        root,
    )
    return merged


async def discover(path: str) -> APISpec:
    """Auto-detect the framework and discover API endpoints.

    The detection strategy is:

    1. If *path* is a directory, delegate to :func:`discover_directory`.
    2. If the file has a ``.json`` / ``.yaml`` / ``.yml`` extension **and**
       contains an ``openapi`` or ``swagger`` top-level key, parse it as an
       OpenAPI spec.
    3. If the file has a ``.py`` extension, parse it as FastAPI source.
    4. For ambiguous JSON/YAML files that do not look like OpenAPI, skip them.

    Parameters
    ----------
    path:
        Filesystem path to a file or directory.

    Returns
    -------
    APISpec
        A populated specification object (may have zero endpoints if nothing
        was found).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    target = Path(path).resolve()
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {target}")

    # Directories get a recursive scan.
    if target.is_dir():
        return await discover_directory(str(target))

    suffix = target.suffix.lower()

    # --- OpenAPI / Swagger specs ---
    if suffix in _OPENAPI_EXTENSIONS:
        text = target.read_text(encoding="utf-8")
        # Peek at the content to decide whether it is truly an OpenAPI doc.
        try:
            if suffix == ".json":
                data = json.loads(text)
            else:
                data = _load_yaml(text)
        except Exception:
            logger.debug("Could not parse %s as JSON/YAML — skipping", target)
            return APISpec(source_path=str(target))

        if isinstance(data, dict) and _is_openapi_content(data):
            return await discover_openapi(str(target))

        logger.debug("%s is JSON/YAML but not an OpenAPI spec — skipping", target)
        return APISpec(source_path=str(target))

    # --- Python source ---
    if suffix == ".py":
        text = target.read_text(encoding="utf-8")

        # Check for MCP server patterns first
        if _source_contains_mcp(text):
            try:
                from agentspec.core.mcp_discovery import discover_mcp
                return await discover_mcp(str(target))
            except Exception as exc:
                logger.warning("MCP discovery failed for %s: %s", target, exc)

        # Flask
        if _source_contains_flask(text):
            try:
                from agentspec.core.flask_discovery import discover_flask
                return await discover_flask(str(target))
            except Exception as exc:
                logger.warning("Flask discovery failed for %s: %s", target, exc)

        # FastAPI (or generic route-decorator pattern)
        if _source_contains_fastapi(text) or re.search(
            r"@\w+\.(get|post|put|patch|delete|head|options)\s*\(", text
        ):
            return await discover_fastapi(str(target))

        logger.debug("%s does not appear to contain Python API routes — skipping", target)
        return APISpec(source_path=str(target))

    # --- JavaScript / TypeScript source ---
    if suffix in _JS_EXTENSIONS:
        text = target.read_text(encoding="utf-8", errors="replace")
        if _source_contains_js_routes(text):
            try:
                from agentspec.core.express_discovery import discover_express
                return await discover_express(str(target))
            except Exception as exc:
                logger.warning("JS/TS discovery failed for %s: %s", target, exc)
        logger.debug("%s does not appear to contain JS/TS routes — skipping", target)
        return APISpec(source_path=str(target))

    logger.debug("Unsupported file type: %s", target)
    return APISpec(source_path=str(target))
