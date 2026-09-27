"""Auto-fix suggestion engine --- analyze test failures and generate actionable fixes.

This is the key differentiator of agentspec: when tests fail, the
:class:`AutoFixer` examines each failure, classifies the root cause, and
produces :class:`FixSuggestion` instances that AI agents can apply directly
to close the build-test-fix loop without human intervention.

Supported fix categories:

* **validation** --- missing input validation (empty strings, negative
  numbers, overlong values, wrong types).
* **security** --- SQL injection, XSS, path traversal, missing auth.
* **error_handling** --- 500 errors where 4xx was expected (missing
  not-found handlers, validation error handlers, etc.).
* **schema** --- response schema mismatches, missing fields, wrong
  content types.
* **performance** --- endpoints that exceed a response-time threshold.

Each suggestion includes framework-specific code, a file hint, and a
plain-English ``agent_instruction`` that can be fed directly to an AI
coding assistant.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from agentspec.models.api import (
    APIEndpoint,
    APISpec,
    FrameworkType,
    HTTPMethod,
    ParamLocation,
    ParamType,
)
from agentspec.models.test import (
    TestCategory,
    TestResult,
    TestSeverity,
    TestStatus,
    TestSuite,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class FixSuggestion(BaseModel):
    """A structured fix suggestion that an agent can apply."""

    id: str
    test_id: str
    title: str
    description: str
    category: str
    severity: str
    endpoint: str
    fix_type: str
    code_suggestion: str
    file_hint: str
    line_hint: int | None = None
    framework_specific: dict[str, Any] = Field(default_factory=dict)
    agent_instruction: str


class FixReport(BaseModel):
    """Collection of fix suggestions for a test run."""

    api_name: str
    total_failures: int
    total_suggestions: int
    suggestions: list[FixSuggestion] = Field(default_factory=list)
    summary: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex for extracting a quoted parameter name from a test name, e.g.
# "Edge case: empty string for 'title' on /tasks" -> "title"
_PARAM_NAME_RE: re.Pattern[str] = re.compile(r"'(\w+)'")

# Regex for extracting the endpoint path from a test name, e.g.
# "... on /tasks/{id}" or "... to POST /tasks"
_ENDPOINT_PATH_RE: re.Pattern[str] = re.compile(r"(?:on|to)\s+(?:GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS\s+)?(/\S+)")

# Regex for extracting method from a test name, e.g. "to POST /tasks"
_ENDPOINT_METHOD_RE: re.Pattern[str] = re.compile(
    r"(?:to|from)\s+(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+/",
)

# Regex for parsing test IDs: {category}_{METHOD}_{slug}_{index}
_TEST_ID_RE: re.Pattern[str] = re.compile(
    r"^(\w+?)_(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)_(.+?)_(\d+)$",
)

# Security payload fragments that indicate specific attack types.
_SQL_INDICATORS: list[str] = ["SQL injection", "sql injection"]
_XSS_INDICATORS: list[str] = ["XSS", "xss"]
_PATH_TRAVERSAL_INDICATORS: list[str] = ["Path traversal", "path traversal"]
_NULL_BYTE_INDICATORS: list[str] = ["Null byte", "null byte"]
_NO_AUTH_INDICATORS: list[str] = ["No auth", "no auth"]

# Database error fragments in response bodies that indicate SQL injection
# succeeded in reaching the query layer.
_DB_ERROR_FRAGMENTS: list[str] = [
    "syntax error",
    "SQL",
    "mysql",
    "sqlite",
    "postgresql",
    "ORA-",
    "SQLSTATE",
]

# Default performance threshold in milliseconds.
_PERFORMANCE_THRESHOLD_MS: float = 5_000.0

# Severity mapping from TestSeverity to the string form used in suggestions.
_SEVERITY_MAP: dict[TestSeverity, str] = {
    TestSeverity.CRITICAL: "critical",
    TestSeverity.HIGH: "high",
    TestSeverity.MEDIUM: "medium",
    TestSeverity.LOW: "low",
    TestSeverity.INFO: "low",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _extract_param_name(test_name: str) -> str | None:
    """Extract a parameter name from a test name string.

    Looks for a single-quoted identifier like ``'title'`` in the test name.
    Returns *None* when no match is found.
    """
    match = _PARAM_NAME_RE.search(test_name)
    return match.group(1) if match else None


def _extract_endpoint_path(test_name: str) -> str | None:
    """Extract the endpoint path from a test name string.

    Matches patterns like ``on /tasks/{id}`` or ``to POST /tasks``.
    """
    match = _ENDPOINT_PATH_RE.search(test_name)
    return match.group(1) if match else None


def _extract_method_from_name(test_name: str) -> str | None:
    """Extract the HTTP method from a test name string."""
    match = _ENDPOINT_METHOD_RE.search(test_name)
    return match.group(1) if match else None


def _parse_test_id(test_id: str) -> dict[str, str] | None:
    """Parse a structured test ID into its components.

    Test IDs follow the pattern ``{category}_{METHOD}_{slug}_{index}``.
    Returns a dict with keys ``category``, ``method``, ``slug``, ``index``
    or *None* if the ID does not match.
    """
    match = _TEST_ID_RE.match(test_id)
    if not match:
        return None
    return {
        "category": match.group(1),
        "method": match.group(2),
        "slug": match.group(3),
        "index": match.group(4),
    }


def _slug_to_path_prefix(slug: str) -> str:
    """Convert a path slug back to a fuzzy path prefix for matching.

    ``users_user_id_posts`` -> ``users`` (first segment) which can be used
    to filter candidate endpoints.
    """
    parts = slug.split("_")
    return parts[0] if parts else ""


def _response_body_str(response_body: Any) -> str:
    """Coerce a response body to a searchable string."""
    if response_body is None:
        return ""
    if isinstance(response_body, str):
        return response_body
    if isinstance(response_body, (dict, list)):
        try:
            import json
            return json.dumps(response_body, default=str)
        except (TypeError, ValueError):
            return str(response_body)
    return str(response_body)


# ---------------------------------------------------------------------------
# Endpoint matching
# ---------------------------------------------------------------------------


def _find_endpoint(
    spec: APISpec,
    *,
    path: str | None = None,
    method: str | None = None,
    slug: str | None = None,
) -> APIEndpoint | None:
    """Find the best-matching endpoint in *spec*.

    Tries exact path + method first, then falls back to slug-based
    substring matching on the endpoint path.
    """
    # Try exact match.
    if path and method:
        try:
            http_method = HTTPMethod(method)
        except ValueError:
            http_method = None
        if http_method:
            ep = spec.get_endpoint(http_method, path)
            if ep is not None:
                return ep

    # Fall back: match path substring.
    if path:
        for ep in spec.endpoints:
            if ep.path == path:
                if method is None:
                    return ep
                try:
                    if ep.method == HTTPMethod(method):
                        return ep
                except ValueError:
                    pass
        # Looser: path contains or is contained.
        for ep in spec.endpoints:
            if path in ep.path or ep.path in path:
                return ep

    # Fall back: slug-based matching.
    if slug:
        prefix = _slug_to_path_prefix(slug)
        for ep in spec.endpoints:
            normalised = ep.path.strip("/").replace("{", "").replace("}", "")
            if prefix and prefix in normalised:
                if method:
                    try:
                        if ep.method == HTTPMethod(method):
                            return ep
                    except ValueError:
                        pass
                else:
                    return ep
        # Even looser: just the prefix.
        for ep in spec.endpoints:
            normalised = ep.path.strip("/").replace("{", "").replace("}", "")
            if prefix and prefix in normalised:
                return ep

    return None


def _endpoint_label(
    endpoint: APIEndpoint | None,
    *,
    fallback_method: str | None = None,
    fallback_path: str | None = None,
) -> str:
    """Build a human-readable endpoint label like ``POST /tasks``."""
    if endpoint:
        return f"{endpoint.method.value} {endpoint.path}"
    parts: list[str] = []
    if fallback_method:
        parts.append(fallback_method)
    if fallback_path:
        parts.append(fallback_path)
    return " ".join(parts) if parts else "unknown endpoint"


def _find_param_type(endpoint: APIEndpoint | None, param_name: str) -> ParamType | None:
    """Look up the declared type of *param_name* on *endpoint*."""
    if endpoint is None:
        return None
    for p in endpoint.parameters:
        if p.name == param_name:
            return p.param_type
    # Check body schema properties.
    if endpoint.request_body_schema:
        props = endpoint.request_body_schema.get("properties", {})
        if param_name in props:
            type_str = props[param_name].get("type", "string")
            mapping = {
                "string": ParamType.STRING,
                "integer": ParamType.INTEGER,
                "number": ParamType.FLOAT,
                "boolean": ParamType.BOOLEAN,
                "array": ParamType.ARRAY,
                "object": ParamType.OBJECT,
            }
            return mapping.get(type_str)
    return None


# ---------------------------------------------------------------------------
# Code-suggestion templates (framework-specific)
# ---------------------------------------------------------------------------


def _fastapi_validation_code(
    param_name: str,
    constraint: str,
    param_type: ParamType | None,
    location: str,
) -> str:
    """Generate a FastAPI/Pydantic validation code snippet."""
    type_hint = "str"
    if param_type == ParamType.INTEGER:
        type_hint = "int"
    elif param_type == ParamType.FLOAT:
        type_hint = "float"
    elif param_type == ParamType.BOOLEAN:
        type_hint = "bool"

    if location == "body":
        return (
            f"# In your Pydantic request model:\n"
            f"{param_name}: {type_hint} = Field(..., {constraint})"
        )
    return (
        f"# In your endpoint function signature:\n"
        f"{param_name}: {type_hint} = Query(..., {constraint})"
    )


def _fastapi_error_handler_code(status_code: int, detail: str) -> str:
    """Generate a FastAPI HTTPException raise snippet."""
    return (
        f"from fastapi import HTTPException\n\n"
        f"raise HTTPException(status_code={status_code}, detail=\"{detail}\")"
    )


def _fastapi_sql_protection_code(param_name: str) -> str:
    """Generate SQL injection protection code."""
    return (
        f"# Use parameterized queries instead of string interpolation:\n"
        f"# BAD:  cursor.execute(f\"SELECT * FROM items WHERE name = '{{{{param}}}}'\")\n"
        f"# GOOD: cursor.execute(\"SELECT * FROM items WHERE name = ?\", ({param_name},))\n"
        f"#\n"
        f"# If using SQLAlchemy:\n"
        f"from sqlalchemy import text\n"
        f"result = session.execute(\n"
        f"    text(\"SELECT * FROM items WHERE name = :name\"),\n"
        f"    {{\"name\": {param_name}}}\n"
        f")"
    )


def _fastapi_xss_protection_code(param_name: str) -> str:
    """Generate XSS protection code."""
    return (
        f"import html\n\n"
        f"# Sanitize output before including in responses:\n"
        f"safe_{param_name} = html.escape({param_name})\n"
        f"#\n"
        f"# Or validate input on receipt:\n"
        f"import re\n"
        f"if re.search(r'<[^>]+>', {param_name}):\n"
        f"    raise HTTPException(status_code=400, detail=\"HTML tags not allowed in '{param_name}'\")"
    )


def _fastapi_path_traversal_code(param_name: str) -> str:
    """Generate path traversal protection code."""
    return (
        f"import os\n\n"
        f"# Validate and sanitize file paths:\n"
        f"safe_path = os.path.basename({param_name})\n"
        f"if '..' in {param_name} or {param_name}.startswith('/'):\n"
        f"    raise HTTPException(status_code=400, detail=\"Invalid path in '{param_name}'\")"
    )


def _fastapi_auth_middleware_code() -> str:
    """Generate auth middleware code."""
    return (
        "from fastapi import Depends, HTTPException, Security\n"
        "from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials\n\n"
        "security = HTTPBearer()\n\n"
        "async def verify_token(\n"
        "    credentials: HTTPAuthorizationCredentials = Security(security),\n"
        ") -> str:\n"
        "    token = credentials.credentials\n"
        "    # Add your token verification logic here\n"
        "    if not token:\n"
        "        raise HTTPException(status_code=401, detail=\"Invalid or missing token\")\n"
        "    return token\n\n"
        "# Add to your endpoint:\n"
        "# @app.get(\"/protected\", dependencies=[Depends(verify_token)])"
    )


def _fastapi_response_model_code(missing_fields: list[str]) -> str:
    """Generate response model fix code for missing fields."""
    fields = "\n".join(
        f"    {field}: str | None = None" for field in missing_fields
    )
    return (
        f"from pydantic import BaseModel\n\n"
        f"# Add missing fields to your response model:\n"
        f"class ResponseModel(BaseModel):\n"
        f"    # ... existing fields ...\n"
        f"{fields}"
    )


def _fastapi_performance_code(endpoint_label: str, duration_ms: float) -> str:
    """Generate performance improvement code."""
    return (
        f"# Endpoint {endpoint_label} is slow ({duration_ms:.0f}ms).\n"
        f"# Consider these optimizations:\n\n"
        f"# 1. Add caching:\n"
        f"from functools import lru_cache\n\n"
        f"# 2. Use async database queries:\n"
        f"# result = await db.fetch_all(query)\n\n"
        f"# 3. Add pagination for list endpoints:\n"
        f"from fastapi import Query\n\n"
        f"async def endpoint(\n"
        f"    skip: int = Query(0, ge=0),\n"
        f"    limit: int = Query(20, ge=1, le=100),\n"
        f"):\n"
        f"    items = await db.fetch_all(query.offset(skip).limit(limit))\n"
        f"    return items"
    )


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def _is_server_error(result: TestResult) -> bool:
    """Return *True* if the response was a 5xx server error."""
    return result.status_code is not None and 500 <= result.status_code < 600


def _expected_client_error(result: TestResult) -> int | None:
    """Infer the expected client-error status code from assertion failures.

    Returns the expected status code (e.g. 400, 404, 422) when the failure
    was a status-code mismatch, or *None* otherwise.
    """
    for failure in result.failures:
        if "status" not in failure.assertion.lower():
            continue
        # Expected field typically looks like "[400, 422]" or "404".
        expected = failure.expected.strip()
        # Try single int.
        try:
            return int(expected)
        except ValueError:
            pass
        # Try list format: extract the first integer.
        match = re.search(r"\d+", expected)
        if match:
            return int(match.group(0))
    return None


def _contains_any(text: str, fragments: list[str]) -> bool:
    """Case-insensitive check: does *text* contain any of *fragments*?"""
    lower = text.lower()
    return any(f.lower() in lower for f in fragments)


# ---------------------------------------------------------------------------
# Per-category analyzers
# ---------------------------------------------------------------------------


def _analyze_validation(
    result: TestResult,
    endpoint: APIEndpoint | None,
    spec: APISpec,
    include_code: bool,
) -> FixSuggestion | None:
    """Analyze an edge-case or error-handling failure for missing validation.

    Detects patterns like:
    - 500 on empty string -> add ``min_length=1``
    - 500 on negative number -> add ``ge=0``
    - 500 on very long string -> add ``max_length``
    - 500 on wrong type -> add type validation
    """
    if not _is_server_error(result):
        return None

    test_name = result.test_name.lower()
    param_name = _extract_param_name(result.test_name)
    ep_path = _extract_endpoint_path(result.test_name)
    ep_label = _endpoint_label(endpoint, fallback_path=ep_path)
    param_type = _find_param_type(endpoint, param_name) if param_name else None
    location = "body"  # default assumption

    if endpoint and param_name:
        for p in endpoint.parameters:
            if p.name == param_name:
                location = "query" if p.location == ParamLocation.QUERY else "body"
                break

    # --- Empty string ---
    if "empty string" in test_name:
        constraint = "min_length=1"
        title = f"Add empty-string validation for '{param_name}' on {ep_label}"
        description = (
            f"The endpoint {ep_label} returns a 500 error when parameter "
            f"'{param_name}' is an empty string. The server should validate "
            f"input and return 400 or 422 instead of crashing."
        )
        code = ""
        if include_code:
            code = _fastapi_validation_code(
                param_name or "field", constraint, param_type, location,
            )
        instruction = (
            f"Add input validation to parameter '{param_name}' on endpoint "
            f"'{ep_label}': use Field(..., min_length=1) to reject empty strings"
        )
        return FixSuggestion(
            id="",  # filled by caller
            test_id=result.test_id,
            title=title,
            description=description,
            category="validation",
            severity=_SEVERITY_MAP.get(result.severity, "medium"),
            endpoint=ep_label,
            fix_type="add_validation",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "Field(..., min_length=1)"),
            agent_instruction=instruction,
        )

    # --- Negative number ---
    if "negative" in test_name:
        constraint = "ge=0"
        title = f"Add non-negative validation for '{param_name}' on {ep_label}"
        description = (
            f"The endpoint {ep_label} returns a 500 error when parameter "
            f"'{param_name}' is a negative number. Add a constraint to ensure "
            f"the value is zero or positive."
        )
        code = ""
        if include_code:
            code = _fastapi_validation_code(
                param_name or "field", constraint, param_type or ParamType.INTEGER, location,
            )
        instruction = (
            f"Add input validation to parameter '{param_name}' on endpoint "
            f"'{ep_label}': use Field(..., ge=0) to reject negative numbers"
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="validation",
            severity=_SEVERITY_MAP.get(result.severity, "medium"),
            endpoint=ep_label,
            fix_type="add_validation",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "Field(..., ge=0)"),
            agent_instruction=instruction,
        )

    # --- Very long string ---
    if "long string" in test_name or "10000 char" in test_name:
        constraint = "max_length=1000"
        title = f"Add max-length validation for '{param_name}' on {ep_label}"
        description = (
            f"The endpoint {ep_label} returns a 500 error when parameter "
            f"'{param_name}' is extremely long (10,000+ characters). Add a "
            f"maximum length constraint to prevent excessive input."
        )
        code = ""
        if include_code:
            code = _fastapi_validation_code(
                param_name or "field", constraint, param_type or ParamType.STRING, location,
            )
        instruction = (
            f"Add input validation to parameter '{param_name}' on endpoint "
            f"'{ep_label}': use Field(..., max_length=1000) to reject excessively "
            f"long strings"
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="validation",
            severity=_SEVERITY_MAP.get(result.severity, "medium"),
            endpoint=ep_label,
            fix_type="add_validation",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "Field(..., max_length=1000)"),
            agent_instruction=instruction,
        )

    # --- Wrong type ---
    if "wrong type" in test_name:
        title = f"Add type validation for '{param_name}' on {ep_label}"
        description = (
            f"The endpoint {ep_label} returns a 500 error when parameter "
            f"'{param_name}' receives a value of the wrong type. Ensure the "
            f"endpoint validates parameter types before processing."
        )
        expected_type = "int" if param_type == ParamType.INTEGER else "str"
        code = ""
        if include_code:
            code = (
                f"# Ensure proper type annotation in your endpoint:\n"
                f"{param_name}: {expected_type}"
            )
        instruction = (
            f"Add type validation for parameter '{param_name}' on endpoint "
            f"'{ep_label}': ensure the parameter has a proper type annotation "
            f"so that invalid types are rejected with 422"
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="validation",
            severity=_SEVERITY_MAP.get(result.severity, "medium"),
            endpoint=ep_label,
            fix_type="add_validation",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, f"type annotation: {expected_type}"),
            agent_instruction=instruction,
        )

    return None


def _analyze_security(
    result: TestResult,
    endpoint: APIEndpoint | None,
    spec: APISpec,
    include_code: bool,
) -> FixSuggestion | None:
    """Analyze a security test failure for vulnerabilities.

    Detects patterns like:
    - SQL injection payload returns 200 with DB error fragments
    - XSS payload reflected in response body
    - Path traversal payload succeeds
    - Unauthenticated request succeeds on protected endpoint
    """
    test_name = result.test_name
    param_name = _extract_param_name(test_name)
    ep_path = _extract_endpoint_path(test_name)
    ep_label = _endpoint_label(endpoint, fallback_path=ep_path)
    response_text = _response_body_str(result.response_body)

    # --- SQL injection ---
    if _contains_any(test_name, _SQL_INDICATORS):
        # Failure means the test detected a problem: either DB error in
        # response or the server accepted the payload without sanitisation.
        has_db_error = _contains_any(response_text, _DB_ERROR_FRAGMENTS)
        title = f"SQL injection vulnerability in '{param_name}' on {ep_label}"
        description = (
            f"The endpoint {ep_label} may be vulnerable to SQL injection "
            f"through parameter '{param_name}'. "
        )
        if has_db_error:
            description += (
                "The response contains database error fragments, indicating "
                "unsanitised input reached the query layer."
            )
        else:
            description += (
                "The injection payload was accepted without proper rejection. "
                "Use parameterised queries to prevent SQL injection."
            )
        code = ""
        if include_code:
            code = _fastapi_sql_protection_code(param_name or "param")
        instruction = (
            f"Add SQL injection protection to endpoint '{ep_label}': "
            f"sanitize string input parameter '{param_name}' before using "
            f"in queries. Switch to parameterised queries or an ORM."
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="security",
            severity="critical",
            endpoint=ep_label,
            fix_type="add_validation",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "parameterised queries"),
            agent_instruction=instruction,
        )

    # --- XSS ---
    if _contains_any(test_name, _XSS_INDICATORS):
        reflected = "<script>" in response_text.lower()
        title = f"XSS vulnerability in '{param_name}' on {ep_label}"
        description = (
            f"The endpoint {ep_label} may be vulnerable to cross-site "
            f"scripting (XSS) through parameter '{param_name}'. "
        )
        if reflected:
            description += (
                "The injected script tag is reflected verbatim in the "
                "response body."
            )
        else:
            description += (
                "The XSS payload was not properly rejected or sanitised."
            )
        code = ""
        if include_code:
            code = _fastapi_xss_protection_code(param_name or "param")
        instruction = (
            f"Add XSS protection to endpoint '{ep_label}': sanitize or "
            f"reject HTML content in parameter '{param_name}'. Use "
            f"html.escape() on output or reject tags on input."
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="security",
            severity="critical",
            endpoint=ep_label,
            fix_type="add_validation",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "html.escape()"),
            agent_instruction=instruction,
        )

    # --- Path traversal ---
    if _contains_any(test_name, _PATH_TRAVERSAL_INDICATORS):
        traversal_success = _contains_any(response_text, ["root:", "/etc/passwd"])
        title = f"Path traversal vulnerability in '{param_name}' on {ep_label}"
        description = (
            f"The endpoint {ep_label} may be vulnerable to path traversal "
            f"through parameter '{param_name}'. "
        )
        if traversal_success:
            description += (
                "The response contains filesystem content, indicating "
                "the traversal payload reached the file system."
            )
        else:
            description += (
                "The path traversal payload was not properly rejected."
            )
        code = ""
        if include_code:
            code = _fastapi_path_traversal_code(param_name or "param")
        instruction = (
            f"Add path traversal protection to endpoint '{ep_label}': "
            f"validate parameter '{param_name}' to reject '..', leading "
            f"slashes, and other path escape sequences."
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="security",
            severity="critical",
            endpoint=ep_label,
            fix_type="add_validation",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "os.path.basename()"),
            agent_instruction=instruction,
        )

    # --- Null byte ---
    if _contains_any(test_name, _NULL_BYTE_INDICATORS):
        title = f"Null byte injection in '{param_name}' on {ep_label}"
        description = (
            f"The endpoint {ep_label} does not properly handle null bytes "
            f"in parameter '{param_name}'. Null bytes can truncate strings "
            f"and bypass validation in some languages and libraries."
        )
        code = ""
        if include_code:
            code = (
                f"# Strip or reject null bytes in input:\n"
                f"if '\\x00' in {param_name or 'param'}:\n"
                f"    raise HTTPException(status_code=400, detail=\"Null bytes not allowed\")\n"
                f"# Or sanitize:\n"
                f"{param_name or 'param'} = {param_name or 'param'}.replace('\\x00', '')"
            )
        instruction = (
            f"Add null byte protection to parameter '{param_name}' on "
            f"endpoint '{ep_label}': strip or reject input containing "
            f"null byte characters (\\x00)."
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="security",
            severity="high",
            endpoint=ep_label,
            fix_type="add_validation",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "null byte stripping"),
            agent_instruction=instruction,
        )

    # --- Missing auth ---
    if _contains_any(test_name, _NO_AUTH_INDICATORS):
        # A failure here means the unauthenticated request was NOT
        # rejected — it returned 200 instead of 401/403.
        title = f"Missing authentication on {ep_label}"
        description = (
            f"The endpoint {ep_label} accepted an unauthenticated request. "
            f"Requests to this endpoint should require valid credentials and "
            f"return 401 or 403 when none are provided."
        )
        code = ""
        if include_code:
            code = _fastapi_auth_middleware_code()
        instruction = (
            f"Add authentication to endpoint '{ep_label}': require a "
            f"valid bearer token or API key. Return 401 when credentials "
            f"are missing and 403 when they are invalid."
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="security",
            severity="critical",
            endpoint=ep_label,
            fix_type="add_auth",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "Depends(verify_token)"),
            agent_instruction=instruction,
        )

    return None


def _analyze_error_handling(
    result: TestResult,
    endpoint: APIEndpoint | None,
    spec: APISpec,
    include_code: bool,
) -> FixSuggestion | None:
    """Analyze an error-handling failure for missing handlers.

    Detects patterns like:
    - 500 instead of 404 -> add not-found handler
    - 500 instead of 422 -> add validation error handler
    - 500 instead of 405 -> add method-not-allowed handler
    - 500 instead of 400 -> add bad-request handler
    """
    if not _is_server_error(result):
        return None

    ep_path = _extract_endpoint_path(result.test_name)
    ep_label = _endpoint_label(endpoint, fallback_path=ep_path)
    expected_status = _expected_client_error(result)

    if expected_status is None:
        # Check if we can infer from the test name.
        test_name_lower = result.test_name.lower()
        if "missing required" in test_name_lower:
            expected_status = 422
        elif "wrong method" in test_name_lower:
            expected_status = 405
        elif "wrong type" in test_name_lower:
            expected_status = 422
        else:
            # Generic 500 -> should be a client error.
            expected_status = 400

    # --- 404 Not Found ---
    if expected_status == 404:
        title = f"Add 404 handler for {ep_label}"
        description = (
            f"The endpoint {ep_label} returns a 500 error instead of 404 "
            f"when the requested resource does not exist. Add a proper "
            f"not-found error handler."
        )
        code = ""
        if include_code:
            code = _fastapi_error_handler_code(404, "Resource not found")
        instruction = (
            f"Add a 404 error handler for endpoint '{ep_label}': return "
            f"HTTPException(status_code=404, detail='Resource not found') "
            f"when the resource doesn't exist"
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="error_handling",
            severity=_SEVERITY_MAP.get(result.severity, "high"),
            endpoint=ep_label,
            fix_type="add_error_handler",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "HTTPException(404)"),
            agent_instruction=instruction,
        )

    # --- 422 Unprocessable Entity ---
    if expected_status == 422:
        param_name = _extract_param_name(result.test_name)
        detail = f"for parameter '{param_name}'" if param_name else ""
        title = f"Add validation error handler {detail} on {ep_label}"
        description = (
            f"The endpoint {ep_label} returns a 500 error instead of 422 "
            f"when invalid data is submitted{' for ' + repr(param_name) if param_name else ''}. "
            f"Add proper request validation to return a 422 with details."
        )
        code = ""
        if include_code:
            code = _fastapi_error_handler_code(
                422,
                f"Validation error{': invalid ' + param_name if param_name else ''}",
            )
        instruction = (
            f"Add a validation error handler for endpoint '{ep_label}': "
            f"validate request data and return HTTPException(status_code=422, "
            f"detail='Validation error') instead of letting the server crash"
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="error_handling",
            severity=_SEVERITY_MAP.get(result.severity, "high"),
            endpoint=ep_label,
            fix_type="add_error_handler",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "HTTPException(422)"),
            agent_instruction=instruction,
        )

    # --- 405 Method Not Allowed ---
    if expected_status == 405:
        title = f"Add method-not-allowed handler for {ep_label}"
        description = (
            f"The endpoint {ep_label} returns a 500 error instead of 405 "
            f"when an unsupported HTTP method is used. This usually "
            f"indicates the framework's default method routing is not "
            f"properly configured."
        )
        code = ""
        if include_code:
            code = _fastapi_error_handler_code(405, "Method not allowed")
        instruction = (
            f"Ensure endpoint '{ep_label}' correctly returns 405 for "
            f"unsupported methods. Check that routes are registered with "
            f"specific methods rather than a catch-all."
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="error_handling",
            severity=_SEVERITY_MAP.get(result.severity, "medium"),
            endpoint=ep_label,
            fix_type="add_error_handler",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "HTTPException(405)"),
            agent_instruction=instruction,
        )

    # --- Generic 400 Bad Request ---
    title = f"Add error handler for {ep_label}"
    description = (
        f"The endpoint {ep_label} returns a 500 server error instead of "
        f"an appropriate {expected_status} client error. Add a try/except "
        f"or input validation to handle the error gracefully."
    )
    code = ""
    if include_code:
        code = _fastapi_error_handler_code(expected_status, "Bad request")
    instruction = (
        f"Add error handling to endpoint '{ep_label}': catch the exception "
        f"that causes the 500 error and return HTTPException(status_code="
        f"{expected_status}) with a descriptive message instead"
    )
    return FixSuggestion(
        id="",
        test_id=result.test_id,
        title=title,
        description=description,
        category="error_handling",
        severity=_SEVERITY_MAP.get(result.severity, "high"),
        endpoint=ep_label,
        fix_type="add_error_handler",
        code_suggestion=code,
        file_hint=spec.source_path,
        framework_specific=_framework_details(spec, f"HTTPException({expected_status})"),
        agent_instruction=instruction,
    )


def _analyze_schema(
    result: TestResult,
    endpoint: APIEndpoint | None,
    spec: APISpec,
    include_code: bool,
) -> FixSuggestion | None:
    """Analyze a schema-validation failure.

    Detects patterns like:
    - Response missing expected fields
    - Wrong content type
    - Response does not match declared schema
    """
    ep_path = _extract_endpoint_path(result.test_name)
    ep_label = _endpoint_label(endpoint, fallback_path=ep_path)
    test_name_lower = result.test_name.lower()

    # --- Content-Type mismatch ---
    if "content-type" in test_name_lower or "content_type" in test_name_lower:
        expected_ct = ""
        actual_ct = ""
        for failure in result.failures:
            if "content" in failure.assertion.lower():
                expected_ct = failure.expected
                actual_ct = failure.actual
                break

        title = f"Fix Content-Type header on {ep_label}"
        description = (
            f"The endpoint {ep_label} returns Content-Type '{actual_ct}' "
            f"but the spec declares '{expected_ct}'. Set the correct "
            f"response content type."
        )
        code = ""
        if include_code:
            code = (
                f"from fastapi.responses import JSONResponse\n\n"
                f"# Set the response class on your endpoint:\n"
                f"@app.get(\"/...\", response_class=JSONResponse)"
            )
        instruction = (
            f"Fix the Content-Type on endpoint '{ep_label}': the response "
            f"should return '{expected_ct}' but currently returns "
            f"'{actual_ct}'. Set response_class or media_type on the route."
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="schema",
            severity=_SEVERITY_MAP.get(result.severity, "high"),
            endpoint=ep_label,
            fix_type="fix_response",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "response_class=JSONResponse"),
            agent_instruction=instruction,
        )

    # --- Schema mismatch / missing fields ---
    missing_fields: list[str] = []
    for failure in result.failures:
        assertion_lower = failure.assertion.lower()
        if "missing" in assertion_lower or "field" in assertion_lower:
            # Try to extract field names from the assertion or expected value.
            fields_match = re.findall(r"'(\w+)'", failure.expected)
            missing_fields.extend(fields_match)
        elif "schema" in assertion_lower:
            fields_match = re.findall(r"'(\w+)'", failure.expected + " " + failure.actual)
            missing_fields.extend(fields_match)

    if missing_fields:
        title = f"Add missing fields to response on {ep_label}"
        field_list = ", ".join(f"'{f}'" for f in missing_fields[:5])
        description = (
            f"The response from {ep_label} is missing expected fields: "
            f"{field_list}. Update the response model or endpoint logic to "
            f"include these fields."
        )
        code = ""
        if include_code:
            code = _fastapi_response_model_code(missing_fields[:5])
        instruction = (
            f"Add missing fields to the response from endpoint '{ep_label}': "
            f"the response should include {field_list}. Update your Pydantic "
            f"response model and ensure the endpoint returns these fields."
        )
        return FixSuggestion(
            id="",
            test_id=result.test_id,
            title=title,
            description=description,
            category="schema",
            severity=_SEVERITY_MAP.get(result.severity, "high"),
            endpoint=ep_label,
            fix_type="fix_schema",
            code_suggestion=code,
            file_hint=spec.source_path,
            framework_specific=_framework_details(spec, "Pydantic response model"),
            agent_instruction=instruction,
        )

    # --- Generic schema failure ---
    title = f"Fix response schema on {ep_label}"
    description = (
        f"The response from {ep_label} does not match the declared schema. "
        f"Review the response model and ensure all fields and types match "
        f"the API specification."
    )
    first_failure = result.failures[0] if result.failures else None
    detail = ""
    if first_failure:
        detail = (
            f" Assertion: {first_failure.assertion}; "
            f"expected: {first_failure.expected}; "
            f"actual: {first_failure.actual}."
        )
    code = ""
    if include_code:
        code = (
            "# Review your response model and ensure it matches the spec.\n"
            "# Check that all declared fields are present and correctly typed."
        )
    instruction = (
        f"Fix the response schema on endpoint '{ep_label}': the response "
        f"does not match the declared API specification.{detail} "
        f"Update the endpoint to return a conforming response."
    )
    return FixSuggestion(
        id="",
        test_id=result.test_id,
        title=title,
        description=description,
        category="schema",
        severity=_SEVERITY_MAP.get(result.severity, "high"),
        endpoint=ep_label,
        fix_type="fix_schema",
        code_suggestion=code,
        file_hint=spec.source_path,
        framework_specific=_framework_details(spec, "response model"),
        agent_instruction=instruction,
    )


def _analyze_performance(
    result: TestResult,
    endpoint: APIEndpoint | None,
    spec: APISpec,
    include_code: bool,
    threshold_ms: float,
) -> FixSuggestion | None:
    """Check if a test result exceeds the performance threshold."""
    if result.duration_ms < threshold_ms:
        return None

    ep_path = _extract_endpoint_path(result.test_name)
    ep_label = _endpoint_label(endpoint, fallback_path=ep_path)

    title = f"Slow endpoint: {ep_label} ({result.duration_ms:.0f}ms)"
    description = (
        f"The endpoint {ep_label} took {result.duration_ms:.0f}ms to "
        f"respond, which exceeds the {threshold_ms:.0f}ms threshold. "
        f"Consider adding pagination, caching, or making database queries "
        f"async to improve response times."
    )
    code = ""
    if include_code:
        code = _fastapi_performance_code(ep_label, result.duration_ms)
    instruction = (
        f"Endpoint '{ep_label}' takes {result.duration_ms:.0f}ms --- "
        f"consider adding pagination, caching, or making database queries "
        f"async to bring response time below {threshold_ms:.0f}ms"
    )
    return FixSuggestion(
        id="",
        test_id=result.test_id,
        title=title,
        description=description,
        category="performance",
        severity="medium" if result.duration_ms < threshold_ms * 2 else "high",
        endpoint=ep_label,
        fix_type="add_rate_limit",
        code_suggestion=code,
        file_hint=spec.source_path,
        framework_specific=_framework_details(spec, "async, caching, pagination"),
        agent_instruction=instruction,
    )


# ---------------------------------------------------------------------------
# Framework details helper
# ---------------------------------------------------------------------------


def _framework_details(spec: APISpec, hint: str) -> dict[str, Any]:
    """Build the ``framework_specific`` dict for a suggestion."""
    details: dict[str, Any] = {
        "framework": spec.framework.value,
        "hint": hint,
    }
    if spec.framework == FrameworkType.FASTAPI:
        details["import"] = "from fastapi import HTTPException, Query"
        details["decorator_style"] = "@app.get / @app.post / etc."
    elif spec.framework == FrameworkType.FLASK:
        details["import"] = "from flask import abort, jsonify"
        details["decorator_style"] = "@app.route(..., methods=[...])"
    return details


# ---------------------------------------------------------------------------
# Summary builder
# ---------------------------------------------------------------------------


def _build_summary(
    api_name: str,
    suggestions: list[FixSuggestion],
    total_failures: int,
) -> str:
    """Build a natural-language summary of fix suggestions for agents.

    The summary is a concise paragraph that an AI agent can parse to
    understand priorities at a glance.
    """
    if not suggestions:
        if total_failures == 0:
            return f"All tests passed for {api_name}. No fixes needed."
        return (
            f"{total_failures} test(s) failed for {api_name}, but no "
            f"actionable fix suggestions could be generated. Review the "
            f"test results manually."
        )

    # Count by category.
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    endpoints_affected: set[str] = set()

    for s in suggestions:
        by_category[s.category] = by_category.get(s.category, 0) + 1
        by_severity[s.severity] = by_severity.get(s.severity, 0) + 1
        endpoints_affected.add(s.endpoint)

    # Build category descriptions.
    parts: list[str] = []
    for cat, count in sorted(by_category.items(), key=lambda x: -x[1]):
        parts.append(f"{count} {cat.replace('_', ' ')}")

    category_str = ", ".join(parts)

    # Build severity breakdown for critical/high.
    critical = by_severity.get("critical", 0)
    high = by_severity.get("high", 0)
    severity_parts: list[str] = []
    if critical:
        severity_parts.append(f"{critical} critical")
    if high:
        severity_parts.append(f"{high} high-severity")
    severity_str = " and ".join(severity_parts) if severity_parts else ""

    # Build priority fixes list.
    priority_fixes = [
        s for s in suggestions if s.severity in ("critical", "high")
    ]
    priority_str = ""
    if priority_fixes:
        priority_items = [s.title for s in priority_fixes[:3]]
        priority_str = (
            f" Priority fixes: {'; '.join(priority_items)}."
        )

    summary = (
        f"Found {len(suggestions)} issue(s) across {len(endpoints_affected)} "
        f"endpoint(s) for {api_name}: {category_str}."
    )
    if severity_str:
        summary += f" Including {severity_str} issue(s)."
    summary += priority_str

    return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class AutoFixer:
    """Analyze test failures and generate actionable fix suggestions.

    The fixer examines each failed :class:`TestResult` in a suite,
    classifies the root cause, and produces :class:`FixSuggestion`
    instances with framework-specific code, file hints, and plain-English
    instructions that an AI coding agent can apply directly.

    Parameters
    ----------
    spec:
        The :class:`APISpec` for the API under test.  Provides endpoint
        metadata, parameter types, and the source file path.
    max_suggestions:
        Maximum number of suggestions to include in the report.  When
        more issues are found, only the highest-severity ones are kept.
    include_code:
        Whether to populate the ``code_suggestion`` field with
        framework-specific snippets.  Set to *False* for a lighter
        report.
    """

    def __init__(
        self,
        spec: APISpec,
        max_suggestions: int = 20,
        include_code: bool = True,
    ) -> None:
        self._spec = spec
        self._max_suggestions = max_suggestions
        self._include_code = include_code
        logger.debug(
            "AutoFixer initialised for '%s' (max_suggestions=%d, include_code=%s)",
            spec.name,
            max_suggestions,
            include_code,
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def analyze(self, suite: TestSuite) -> FixReport:
        """Analyze all test failures and generate fix suggestions.

        Iterates over every non-passing result in *suite*, delegates to
        :meth:`suggest_for_result` for analysis, and assembles a
        :class:`FixReport` with a natural-language summary.

        Parameters
        ----------
        suite:
            A completed :class:`TestSuite` with populated results.

        Returns
        -------
        FixReport
            A report containing up to ``max_suggestions`` fix suggestions,
            ordered by severity (critical first).
        """
        logger.info(
            "Analyzing test suite for '%s' (%d result(s), %d failure(s))",
            suite.api_name,
            suite.total,
            suite.failed + suite.errors,
        )

        all_suggestions: list[FixSuggestion] = []
        suggestion_counter = 0

        for result in suite.results:
            if result.status == TestStatus.PASSED:
                continue
            if result.status == TestStatus.SKIPPED:
                continue

            suggestions = self.suggest_for_result(result)
            for suggestion in suggestions:
                suggestion.id = f"fix_{suggestion_counter}_{suggestion.category}"
                suggestion_counter += 1
                all_suggestions.append(suggestion)

        # Sort by severity: critical > high > medium > low.
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        all_suggestions.sort(key=lambda s: severity_order.get(s.severity, 99))

        # Trim to max.
        trimmed = all_suggestions[: self._max_suggestions]

        total_failures = suite.failed + suite.errors
        summary = _build_summary(suite.api_name, trimmed, total_failures)

        report = FixReport(
            api_name=suite.api_name,
            total_failures=total_failures,
            total_suggestions=len(trimmed),
            suggestions=trimmed,
            summary=summary,
        )

        logger.info(
            "Generated %d fix suggestion(s) from %d failure(s) for '%s'",
            len(trimmed),
            total_failures,
            suite.api_name,
        )
        return report

    def suggest_for_result(self, result: TestResult) -> list[FixSuggestion]:
        """Generate fix suggestions for a single failed test result.

        Dispatches to category-specific analyzers based on the test's
        :attr:`~TestResult.category` and failure characteristics.  A
        single result may yield zero or more suggestions (zero when the
        failure does not match a known pattern).

        Parameters
        ----------
        result:
            A :class:`TestResult` with status ``FAILED`` or ``ERROR``.

        Returns
        -------
        list[FixSuggestion]
            Zero or more suggestions.  Each has a placeholder ``id``
            that :meth:`analyze` replaces with a unique identifier.
        """
        if result.status in (TestStatus.PASSED, TestStatus.SKIPPED):
            return []

        suggestions: list[FixSuggestion] = []

        # Resolve the endpoint from the test metadata.
        parsed_id = _parse_test_id(result.test_id)
        ep_path = _extract_endpoint_path(result.test_name)
        ep_method = (
            _extract_method_from_name(result.test_name)
            or (parsed_id["method"] if parsed_id else None)
        )
        slug = parsed_id["slug"] if parsed_id else None

        endpoint = _find_endpoint(
            self._spec,
            path=ep_path,
            method=ep_method,
            slug=slug,
        )

        # --- Dispatch to category-specific analyzers ---

        if result.category == TestCategory.EDGE_CASE:
            suggestion = _analyze_validation(
                result, endpoint, self._spec, self._include_code,
            )
            if suggestion is not None:
                suggestions.append(suggestion)

        elif result.category == TestCategory.SECURITY:
            suggestion = _analyze_security(
                result, endpoint, self._spec, self._include_code,
            )
            if suggestion is not None:
                suggestions.append(suggestion)

        elif result.category == TestCategory.ERROR_HANDLING:
            # Error handling failures might be validation issues OR
            # missing error handlers.  Try both.
            validation_suggestion = _analyze_validation(
                result, endpoint, self._spec, self._include_code,
            )
            if validation_suggestion is not None:
                suggestions.append(validation_suggestion)
            else:
                handler_suggestion = _analyze_error_handling(
                    result, endpoint, self._spec, self._include_code,
                )
                if handler_suggestion is not None:
                    suggestions.append(handler_suggestion)

        elif result.category == TestCategory.SCHEMA_VALIDATION:
            suggestion = _analyze_schema(
                result, endpoint, self._spec, self._include_code,
            )
            if suggestion is not None:
                suggestions.append(suggestion)

        elif result.category == TestCategory.PERFORMANCE:
            suggestion = _analyze_performance(
                result,
                endpoint,
                self._spec,
                self._include_code,
                _PERFORMANCE_THRESHOLD_MS,
            )
            if suggestion is not None:
                suggestions.append(suggestion)

        # --- Cross-cutting: performance check on any slow test ---
        if result.category != TestCategory.PERFORMANCE:
            perf_suggestion = _analyze_performance(
                result,
                endpoint,
                self._spec,
                self._include_code,
                _PERFORMANCE_THRESHOLD_MS,
            )
            if perf_suggestion is not None:
                suggestions.append(perf_suggestion)

        return suggestions
