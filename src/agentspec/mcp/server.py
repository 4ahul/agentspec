"""MCP (Model Context Protocol) server for agentspec.

Exposes agentspec's API discovery, testing, registration, and compatibility
checking capabilities as MCP tools so that AI agents (Claude Code, Codex,
Cursor, etc.) can use them during their workflow.

Tools
-----
- **test_api** -- Discover and test an API, returning a results summary.
- **discover_api** -- Discover API endpoints from source code or an OpenAPI spec.
- **register_api** -- Discover, test, and register an API in the local registry.
- **list_registered** -- List all APIs in the registry.
- **check_compatibility** -- Check compatibility between two registered APIs.

The server communicates over stdio using the MCP SDK's ``stdio_server``
transport.  Start it with ``agentspec serve`` or directly via
``python -m agentspec.mcp.server``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import traceback
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from agentspec.core.discovery import discover
from agentspec.core.generator import TestGenerator
from agentspec.core.runner import TestRunner
from agentspec.models.test import TestCategory, TestStatus
from agentspec.registry.store import RegistryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

server = Server("agentspec")

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_TOOLS: list[Tool] = [
    Tool(
        name="test_api",
        description=(
            "Discover API endpoints from source code or an OpenAPI spec, "
            "auto-generate a comprehensive test suite (happy path, edge cases, "
            "error handling, security, schema validation), execute the tests "
            "against a live server, and return a summary of results."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Filesystem path to API source code (.py file), "
                        "an OpenAPI/Swagger spec (.json/.yaml), or a directory "
                        "to scan recursively."
                    ),
                },
                "base_url": {
                    "type": "string",
                    "description": (
                        "Base URL of the running API server to test against "
                        "(e.g. 'http://localhost:8000')."
                    ),
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of test categories to run. "
                        "Valid values: happy_path, edge_case, error_handling, "
                        "security, schema_validation, performance. "
                        "Omit to run all categories."
                    ),
                },
            },
            "required": ["path", "base_url"],
        },
    ),
    Tool(
        name="discover_api",
        description=(
            "Discover API endpoints from source code or an OpenAPI/Swagger "
            "specification. Returns structured information about each endpoint "
            "including method, path, parameters, and authentication requirements."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Filesystem path to a Python source file, OpenAPI spec, "
                        "or directory to scan."
                    ),
                },
            },
            "required": ["path"],
        },
    ),
    Tool(
        name="register_api",
        description=(
            "Discover, test, and register an API in the local agentspec "
            "registry. The API must pass at least some tests to be registered. "
            "Returns the registration details including ID, pass rate, and "
            "endpoint count."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Filesystem path to API source code, OpenAPI spec, "
                        "or directory."
                    ),
                },
                "base_url": {
                    "type": "string",
                    "description": (
                        "Base URL of the running API server to test against."
                    ),
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Override the API name. Defaults to the name discovered "
                        "from the source."
                    ),
                },
                "version": {
                    "type": "string",
                    "description": (
                        "Override the API version. Defaults to the version "
                        "discovered from the source."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional tags for categorizing the registered API."
                    ),
                },
            },
            "required": ["path", "base_url"],
        },
    ),
    Tool(
        name="list_registered",
        description=(
            "List all APIs currently registered in the local agentspec "
            "registry. Returns each API's ID, name, version, endpoint count, "
            "pass rate, and last tested timestamp."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="check_compatibility",
        description=(
            "Check compatibility between two registered APIs. Compares their "
            "endpoints, parameters, request bodies, and response schemas. "
            "Returns a compatibility score, pass/fail verdict, and a list of "
            "specific issues found."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "api_a_id": {
                    "type": "string",
                    "description": "Registry ID of the first API.",
                },
                "api_b_id": {
                    "type": "string",
                    "description": "Registry ID of the second API.",
                },
            },
            "required": ["api_a_id", "api_b_id"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Tool listing handler
# ---------------------------------------------------------------------------


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of tools exposed by this MCP server."""
    return _TOOLS


# ---------------------------------------------------------------------------
# Tool execution handler
# ---------------------------------------------------------------------------


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch an incoming tool call to the appropriate handler.

    Every handler returns a human-readable text summary suitable for an AI
    agent to interpret and act on.  Errors are caught and returned as
    descriptive text rather than crashing the server.
    """
    logger.info("Tool called: %s with arguments: %s", name, arguments)

    try:
        if name == "test_api":
            text = await _handle_test_api(arguments)
        elif name == "discover_api":
            text = await _handle_discover_api(arguments)
        elif name == "register_api":
            text = await _handle_register_api(arguments)
        elif name == "list_registered":
            text = await _handle_list_registered(arguments)
        elif name == "check_compatibility":
            text = await _handle_check_compatibility(arguments)
        else:
            text = f"Error: Unknown tool '{name}'."
    except Exception as exc:
        logger.error("Tool '%s' failed: %s", name, exc, exc_info=True)
        text = (
            f"Error executing tool '{name}': {exc}\n\n"
            f"Traceback:\n{traceback.format_exc()}"
        )

    return [TextContent(type="text", text=text)]


# ---------------------------------------------------------------------------
# Handler: test_api
# ---------------------------------------------------------------------------


def _parse_categories(raw: list[str] | None) -> list[TestCategory] | None:
    """Parse a list of category name strings into ``TestCategory`` enum values.

    Returns ``None`` (meaning all categories) when *raw* is ``None`` or empty.
    Raises ``ValueError`` for unrecognised names.
    """
    if not raw:
        return None

    valid = {c.value: c for c in TestCategory}
    categories: list[TestCategory] = []
    for name in raw:
        normalized = name.strip().lower()
        if normalized not in valid:
            raise ValueError(
                f"Unknown test category '{name}'. "
                f"Valid categories: {', '.join(sorted(valid))}"
            )
        categories.append(valid[normalized])
    return categories or None


async def _handle_test_api(arguments: dict[str, Any]) -> str:
    """Discover an API, generate tests, run them, and return a summary."""
    path: str = arguments["path"]
    base_url: str = arguments["base_url"]
    raw_categories: list[str] | None = arguments.get("categories")

    categories = _parse_categories(raw_categories)

    # 1. Discover
    spec = await discover(path)
    if not spec.endpoints:
        return (
            f"No API endpoints discovered at path: {path}\n"
            "Ensure the path points to a FastAPI source file, "
            "an OpenAPI spec, or a directory containing them."
        )

    # 2. Generate tests
    generator = TestGenerator(categories=categories)
    suite = generator.generate(spec)

    # 3. Run tests
    runner = TestRunner(base_url=base_url)
    suite = await runner.run(suite)

    # 4. Format results
    lines: list[str] = [
        "=== agentspec Test Results ===",
        "",
        f"API:        {spec.name} (v{spec.version})",
        f"Source:     {path}",
        f"Base URL:   {base_url}",
        f"Framework:  {spec.framework.value}",
        "",
        f"Endpoints:  {spec.endpoint_count}",
        f"Tests:      {suite.total}",
        f"Passed:     {suite.passed}",
        f"Failed:     {suite.failed}",
        f"Errors:     {suite.errors}",
        f"Skipped:    {suite.skipped}",
        f"Pass rate:  {suite.pass_rate:.1f}%",
        f"Duration:   {suite.duration_ms:.0f} ms",
    ]

    # List failures
    failed_results = [
        r for r in suite.results
        if r.status in (TestStatus.FAILED, TestStatus.ERROR)
    ]
    if failed_results:
        lines.append("")
        lines.append(f"--- Failures ({len(failed_results)}) ---")
        for result in failed_results:
            lines.append("")
            lines.append(f"  [{result.status.value.upper()}] {result.test_name}")
            lines.append(f"    Category: {result.category.value}")
            lines.append(f"    Severity: {result.severity.value}")
            if result.status_code is not None:
                lines.append(f"    Status code: {result.status_code}")
            if result.error_message:
                lines.append(f"    Error: {result.error_message}")
            for failure in result.failures:
                lines.append(
                    f"    - {failure.assertion}: "
                    f"expected {failure.expected}, got {failure.actual}"
                )
    else:
        lines.append("")
        lines.append("All tests passed!")

    # Summary by category
    by_cat = suite.by_category()
    if by_cat:
        lines.append("")
        lines.append("--- Results by Category ---")
        for cat, results in sorted(by_cat.items(), key=lambda x: x[0].value):
            cat_passed = sum(1 for r in results if r.status == TestStatus.PASSED)
            cat_total = len(results)
            cat_rate = (cat_passed / cat_total * 100) if cat_total else 0.0
            lines.append(
                f"  {cat.value:25s}  {cat_passed}/{cat_total}  ({cat_rate:.0f}%)"
            )

    # Auto-fix suggestions
    if failed_results:
        try:
            from agentspec.core.fixer import AutoFixer

            fixer = AutoFixer(spec=spec)
            fix_report = fixer.analyze(suite)
            if fix_report.suggestions:
                lines.append("")
                lines.append("=== Fix Suggestions ===")
                lines.append("")
                lines.append(fix_report.summary)
                lines.append("")
                for fix in fix_report.suggestions[:15]:
                    lines.append(f"[{fix.severity.upper()}] {fix.title}")
                    lines.append(f"  Endpoint: {fix.endpoint}")
                    lines.append(f"  {fix.description}")
                    if fix.code_suggestion:
                        lines.append(f"  Code fix:")
                        for code_line in fix.code_suggestion.split("\n")[:6]:
                            lines.append(f"    {code_line}")
                    lines.append(f"  Action: {fix.agent_instruction}")
                    lines.append("")
        except Exception as exc:
            logger.warning("Auto-fix analysis failed: %s", exc)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handler: discover_api
# ---------------------------------------------------------------------------


async def _handle_discover_api(arguments: dict[str, Any]) -> str:
    """Discover API endpoints and return structured information."""
    path: str = arguments["path"]

    spec = await discover(path)

    if not spec.endpoints:
        return (
            f"No API endpoints discovered at path: {path}\n"
            "Ensure the path points to a FastAPI source file, "
            "an OpenAPI spec, or a directory containing them."
        )

    lines: list[str] = [
        "=== Discovered API ===",
        "",
        f"Name:       {spec.name}",
        f"Version:    {spec.version}",
        f"Framework:  {spec.framework.value}",
        f"Source:     {spec.source_path}",
    ]
    if spec.description:
        lines.append(f"Description: {spec.description}")
    if spec.base_url:
        lines.append(f"Base URL:   {spec.base_url}")
    if spec.auth_schemes:
        lines.append(f"Auth:       {', '.join(spec.auth_schemes)}")

    lines.append(f"Endpoints:  {spec.endpoint_count}")
    lines.append("")
    lines.append("--- Endpoints ---")

    for ep in spec.endpoints:
        auth_marker = " [AUTH]" if ep.requires_auth else ""
        deprecated_marker = " [DEPRECATED]" if ep.deprecated else ""
        lines.append(
            f"  {ep.method.value:7s} {ep.path}{auth_marker}{deprecated_marker}"
        )
        if ep.summary:
            lines.append(f"          Summary: {ep.summary}")

        # Parameters
        if ep.parameters:
            param_parts: list[str] = []
            for p in ep.parameters:
                req = "*" if p.required else ""
                param_parts.append(
                    f"{p.name}{req} ({p.param_type.value}, {p.location.value})"
                )
            lines.append(f"          Params:  {', '.join(param_parts)}")

        # Request body
        if ep.request_body_schema:
            props = ep.request_body_schema.get("properties", {})
            if props:
                field_names = list(props.keys())
                lines.append(
                    f"          Body:    {{{', '.join(field_names)}}}"
                )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handler: register_api
# ---------------------------------------------------------------------------


async def _handle_register_api(arguments: dict[str, Any]) -> str:
    """Discover, test, register an API, and return the registration details."""
    path: str = arguments["path"]
    base_url: str = arguments["base_url"]
    name_override: str | None = arguments.get("name")
    version_override: str | None = arguments.get("version")
    tags: list[str] | None = arguments.get("tags")

    # 1. Discover
    spec = await discover(path)
    if not spec.endpoints:
        return (
            f"No API endpoints discovered at path: {path}\n"
            "Cannot register an API with zero endpoints."
        )

    if name_override:
        spec.name = name_override
    if version_override:
        spec.version = version_override

    # 2. Generate and run tests
    generator = TestGenerator()
    suite = generator.generate(spec)
    runner = TestRunner(base_url=base_url)
    suite = await runner.run(suite)

    pass_rate = suite.pass_rate / 100.0  # Normalize to 0.0 - 1.0

    if pass_rate <= 0.0:
        return (
            f"All tests failed for {spec.name} (v{spec.version}).\n"
            f"Pass rate: 0.0%\n"
            f"The API cannot be registered with a 0% pass rate.\n"
            f"Fix the failing tests and try again."
        )

    # 3. Register
    store = RegistryStore()
    try:
        entry = store.register(spec=spec, pass_rate=pass_rate, tags=tags or [])
    except ValueError as exc:
        return f"Registration failed: {exc}"

    lines: list[str] = [
        "=== API Registered ===",
        "",
        f"ID:         {entry.id}",
        f"Name:       {entry.name}",
        f"Version:    {entry.version}",
        f"Endpoints:  {spec.endpoint_count}",
        f"Tests run:  {suite.total}",
        f"Pass rate:  {suite.pass_rate:.1f}%",
    ]
    if entry.tags:
        lines.append(f"Tags:       {', '.join(entry.tags)}")
    lines.append(f"Registered: {entry.registered_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handler: list_registered
# ---------------------------------------------------------------------------


async def _handle_list_registered(arguments: dict[str, Any]) -> str:
    """List all registered APIs in a table-like text format."""
    store = RegistryStore()
    entries = store.list_all()

    if not entries:
        return "No APIs registered in the agentspec registry."

    lines: list[str] = [
        f"=== Registered APIs ({len(entries)}) ===",
        "",
        f"{'ID':<30s}  {'Name':<25s}  {'Version':<10s}  "
        f"{'Endpoints':>9s}  {'Pass Rate':>10s}  {'Last Tested':<20s}",
        "-" * 115,
    ]

    for entry in entries:
        last_tested = (
            entry.last_tested_at.strftime("%Y-%m-%d %H:%M")
            if entry.last_tested_at
            else "-"
        )
        rate_str = f"{entry.last_test_pass_rate * 100:.1f}%"
        lines.append(
            f"{entry.id:<30s}  {entry.name:<25s}  {entry.version:<10s}  "
            f"{entry.spec.endpoint_count:>9d}  {rate_str:>10s}  {last_tested:<20s}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Handler: check_compatibility
# ---------------------------------------------------------------------------


async def _handle_check_compatibility(arguments: dict[str, Any]) -> str:
    """Check compatibility between two registered APIs."""
    api_a_id: str = arguments["api_a_id"]
    api_b_id: str = arguments["api_b_id"]

    store = RegistryStore()

    try:
        result = store.check_compatibility(api_a_id, api_b_id)
    except KeyError as exc:
        return f"Compatibility check failed: {exc}"

    score_pct = result.score * 100
    verdict = "COMPATIBLE" if result.compatible else "INCOMPATIBLE"

    lines: list[str] = [
        "=== Compatibility Check ===",
        "",
        f"API A:       {result.api_a_name} ({result.api_a_id})",
        f"API B:       {result.api_b_name} ({result.api_b_id})",
        f"Score:       {score_pct:.1f}%",
        f"Verdict:     {verdict}",
    ]

    if result.issues:
        lines.append("")
        lines.append(f"--- Issues ({len(result.issues)}) ---")
        for issue in result.issues:
            endpoint_str = f" [{issue.endpoint}]" if issue.endpoint else ""
            lines.append(
                f"  [{issue.severity.upper():6s}] {issue.issue_type}"
                f"{endpoint_str}: {issue.description}"
            )
    else:
        lines.append("")
        lines.append("No compatibility issues found.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


async def run_server() -> None:
    """Start the MCP server on stdio.

    This coroutine blocks until the transport is closed (e.g. the parent
    process disconnects or sends EOF).
    """
    logger.info("Starting agentspec MCP server on stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
    logger.info("agentspec MCP server stopped")


def run(*, host: str = "localhost", port: int = 8080) -> None:
    """Entry point called by the CLI ``serve`` command.

    The *host* and *port* parameters are accepted for forward-compatibility
    with future HTTP/SSE transports but are currently unused -- the server
    communicates exclusively over stdio.

    Parameters
    ----------
    host:
        Reserved for future HTTP transport.
    port:
        Reserved for future HTTP transport.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    logger.info(
        "run() called (host=%s, port=%d) -- using stdio transport", host, port
    )
    asyncio.run(run_server())


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run()
