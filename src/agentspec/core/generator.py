"""Test generation engine --- automatically produce comprehensive test suites from API specs.

This is the core innovation of agentspec: given a discovered :class:`APISpec`, the
:class:`TestGenerator` creates a broad set of :class:`TestCase` instances covering
happy-path behaviour, edge cases, error handling, security probes, and schema
validation.  The generated suite is deterministic (no randomness) and can be
executed by the runner without further human configuration.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from agentspec.models.api import (
    APIEndpoint,
    APIParameter,
    APISpec,
    HTTPMethod,
    ParamLocation,
    ParamType,
)
from agentspec.models.test import (
    TestCase,
    TestCategory,
    TestSeverity,
    TestSuite,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PATH_PARAM_RE: re.Pattern[str] = re.compile(r"\{(\w+)\}")

# Methods that conventionally carry a request body.
_BODY_METHODS: frozenset[HTTPMethod] = frozenset({
    HTTPMethod.POST,
    HTTPMethod.PUT,
    HTTPMethod.PATCH,
})

# All valid methods for a "wrong method" test --- lightweight methods only.
_ALL_METHODS: list[HTTPMethod] = [
    HTTPMethod.GET,
    HTTPMethod.POST,
    HTTPMethod.PUT,
    HTTPMethod.PATCH,
    HTTPMethod.DELETE,
]

# ---------------------------------------------------------------------------
# Smart value generation
# ---------------------------------------------------------------------------

# Mapping of common parameter name patterns to plausible test values.  The keys
# are checked with ``in`` against the lowercased parameter name so that both
# ``user_email`` and ``email_address`` match ``email``.
_NAME_HINTS: list[tuple[str, Any]] = [
    ("email", "test@example.com"),
    ("phone", "+15555555555"),
    ("url", "https://example.com"),
    ("uri", "https://example.com"),
    ("website", "https://example.com"),
    ("username", "testuser"),
    ("user_name", "testuser"),
    ("first_name", "Alice"),
    ("last_name", "Smith"),
    ("full_name", "Alice Smith"),
    ("password", "T3stP@ssw0rd!"),
    ("token", "tok_test_abc123"),
    ("api_key", "key_test_abc123"),
    ("address", "123 Test Street"),
    ("city", "Testville"),
    ("state", "CA"),
    ("country", "US"),
    ("zip", "90210"),
    ("postal", "90210"),
    ("title", "Test Title"),
    ("description", "A test description"),
    ("content", "Test content body"),
    ("message", "Hello, this is a test message"),
    ("comment", "This is a test comment"),
    ("slug", "test-slug"),
    ("status", "active"),
    ("role", "user"),
    ("type", "default"),
    ("category", "general"),
    ("tag", "test"),
    ("color", "#ff0000"),
    ("date", "2025-01-15"),
    ("time", "12:00:00"),
    ("datetime", "2025-01-15T12:00:00Z"),
    ("timestamp", "2025-01-15T12:00:00Z"),
    ("age", 25),
    ("count", 10),
    ("quantity", 1),
    ("amount", 100),
    ("price", 9.99),
    ("lat", 37.7749),
    ("latitude", 37.7749),
    ("lng", -122.4194),
    ("lon", -122.4194),
    ("longitude", -122.4194),
    ("page", 1),
    ("per_page", 20),
    ("page_size", 20),
    ("limit", 20),
    ("offset", 0),
    ("sort", "created_at"),
    ("order", "asc"),
    ("query", "test search"),
    ("search", "test"),
    ("q", "test"),
    ("filter", "all"),
    ("locale", "en-US"),
    ("language", "en"),
    ("lang", "en"),
    ("currency", "USD"),
    ("format", "json"),
]

# Fallback values keyed by ParamType when no name hint matches.
_TYPE_DEFAULTS: dict[ParamType, Any] = {
    ParamType.STRING: "test_string",
    ParamType.INTEGER: 1,
    ParamType.FLOAT: 1.0,
    ParamType.BOOLEAN: True,
    ParamType.ARRAY: ["item1", "item2"],
    ParamType.OBJECT: {"key": "value"},
    ParamType.FILE: "dummy_file_content",
    ParamType.ANY: "test",
}


def _smart_value(name: str, param_type: ParamType) -> Any:
    """Return a deterministic, plausible test value for a parameter.

    First checks if the parameter *name* matches a known pattern (e.g. names
    containing ``email`` yield ``test@example.com``).  Falls back to a
    type-based default.
    """
    lower = name.lower()

    # Special case: any name containing "id" that looks like an identifier.
    if lower == "id" or lower.endswith("_id") or lower.endswith("id"):
        if param_type in (ParamType.INTEGER, ParamType.FLOAT):
            return 1
        return "test-id-1"

    # Special case: priority-like enums common in task/issue trackers
    if lower == "priority":
        return "medium"

    # Walk the hint table looking for a substring match.
    for hint_key, hint_value in _NAME_HINTS:
        if hint_key in lower:
            return hint_value

    return _TYPE_DEFAULTS.get(param_type, "test")


def _wrong_type_value(param_type: ParamType) -> Any:
    """Return a value whose type conflicts with the expected *param_type*.

    Used by error-handling tests to verify type validation.
    """
    match param_type:
        case ParamType.INTEGER | ParamType.FLOAT:
            return "not_a_number"
        case ParamType.BOOLEAN:
            return "not_a_boolean"
        case ParamType.STRING:
            return 99999
        case ParamType.ARRAY:
            return "not_an_array"
        case ParamType.OBJECT:
            return "not_an_object"
        case _:
            return object()  # deliberately odd type


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _slugify_path(path: str) -> str:
    """Convert an endpoint path to a slug suitable for test-case IDs.

    ``/users/{user_id}/posts`` becomes ``users_user_id_posts``.
    """
    slug = path.strip("/")
    slug = slug.replace("{", "").replace("}", "")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
    slug = slug.strip("_").lower()
    return slug or "root"


def _fill_path_params(
    path: str,
    params: list[APIParameter],
    *,
    overrides: dict[str, Any] | None = None,
    method: HTTPMethod | None = None,
) -> tuple[str, dict[str, Any]]:
    """Substitute ``{param}`` placeholders in *path* with test values.

    Returns ``(resolved_path, url_params_dict)``.  The *url_params_dict* maps
    each path parameter name to the value used, which the :class:`TestCase`
    stores so that runners that perform their own substitution have access to
    the raw values.

    When *method* is given, ID-like parameters use a different value per method
    to prevent concurrent test execution conflicts (e.g. DELETE removing a
    resource that PUT needs).
    """
    url_params: dict[str, Any] = {}
    overrides = overrides or {}

    # Assign different IDs per method to avoid concurrent conflicts.
    # GET→1, PUT/PATCH→2, DELETE→3, POST→4, others→1
    _METHOD_ID_OFFSET: dict[HTTPMethod, int] = {
        HTTPMethod.GET: 1,
        HTTPMethod.HEAD: 1,
        HTTPMethod.PUT: 2,
        HTTPMethod.PATCH: 2,
        HTTPMethod.DELETE: 3,
        HTTPMethod.POST: 4,
        HTTPMethod.OPTIONS: 1,
    }
    id_value = _METHOD_ID_OFFSET.get(method, 1) if method else 1

    param_lookup: dict[str, APIParameter] = {
        p.name: p
        for p in params
        if p.location == ParamLocation.PATH
    }

    def _replacer(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in overrides:
            val = overrides[name]
        elif name in param_lookup:
            p = param_lookup[name]
            lower = p.name.lower()
            # Use method-specific ID for id-like path params
            if lower == "id" or lower.endswith("_id") or lower.endswith("id"):
                if p.param_type in (ParamType.INTEGER, ParamType.FLOAT):
                    val = id_value
                else:
                    val = f"test-id-{id_value}"
            else:
                val = _smart_value(p.name, p.param_type)
        else:
            val = "test-value"
        url_params[name] = val
        return str(val)

    resolved = _PATH_PARAM_RE.sub(_replacer, path)
    return resolved, url_params


# ---------------------------------------------------------------------------
# Body generation
# ---------------------------------------------------------------------------


def _generate_body_from_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Produce a plausible JSON body from a JSON-Schema-like *schema* dict.

    Handles the ``properties`` / ``required`` structure emitted by both OpenAPI
    specs and the agentspec FastAPI discovery engine.
    """
    if not schema:
        return None

    schema_type = schema.get("type", "object")
    if schema_type != "object":
        return None

    properties: dict[str, Any] = schema.get("properties", {})
    if not properties:
        return None

    body: dict[str, Any] = {}
    for field_name, field_def in properties.items():
        field_type_str = field_def.get("type", "string")

        # Map the schema type string to ParamType for smart value lookup.
        type_map: dict[str, ParamType] = {
            "string": ParamType.STRING,
            "integer": ParamType.INTEGER,
            "number": ParamType.FLOAT,
            "float": ParamType.FLOAT,
            "boolean": ParamType.BOOLEAN,
            "array": ParamType.ARRAY,
            "object": ParamType.OBJECT,
        }
        pt = type_map.get(field_type_str, ParamType.STRING)

        # Use enum values if available (common in FastAPI Pydantic models).
        enum_values = field_def.get("enum")
        if enum_values and isinstance(enum_values, list) and len(enum_values) > 0:
            body[field_name] = enum_values[0]
        elif "default" in field_def and field_def["default"] is not None:
            body[field_name] = field_def["default"]
        else:
            body[field_name] = _smart_value(field_name, pt)

        # Ensure array-typed fields actually produce lists.
        if pt == ParamType.ARRAY and not isinstance(body[field_name], list):
            body[field_name] = [body[field_name]]

    return body


def _generate_empty_body_from_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    """Produce a body with only the required fields filled, optionals omitted."""
    if not schema:
        return None

    properties: dict[str, Any] = schema.get("properties", {})
    required_fields: list[str] = schema.get("required", [])
    if not properties:
        return None

    body: dict[str, Any] = {}
    for field_name, field_def in properties.items():
        if field_name not in required_fields:
            continue
        field_type_str = field_def.get("type", "string")
        type_map: dict[str, ParamType] = {
            "string": ParamType.STRING,
            "integer": ParamType.INTEGER,
            "number": ParamType.FLOAT,
            "float": ParamType.FLOAT,
            "boolean": ParamType.BOOLEAN,
            "array": ParamType.ARRAY,
            "object": ParamType.OBJECT,
        }
        pt = type_map.get(field_type_str, ParamType.STRING)
        value = _smart_value(field_name, pt)
        if pt == ParamType.ARRAY and not isinstance(value, list):
            value = [value]
        body[field_name] = value

    return body if body else None


# ---------------------------------------------------------------------------
# Query / header param helpers
# ---------------------------------------------------------------------------


def _build_query_params(
    params: list[APIParameter],
    *,
    include_optional: bool = False,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a query-parameter dict from an endpoint's parameter list."""
    overrides = overrides or {}
    result: dict[str, Any] = {}
    for p in params:
        if p.location != ParamLocation.QUERY:
            continue
        if not p.required and not include_optional:
            continue
        if p.name in overrides:
            result[p.name] = overrides[p.name]
        else:
            # Prefer enum values from constraints if available — the smart
            # value heuristic may return a value not in the accepted set.
            enum_vals = p.constraints.get("enum_values")
            if enum_vals and isinstance(enum_vals, list) and len(enum_vals) > 0:
                result[p.name] = enum_vals[0]
            else:
                result[p.name] = _smart_value(p.name, p.param_type)
    return result


def _build_headers(
    params: list[APIParameter],
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Build a headers dict from an endpoint's header parameters."""
    overrides = overrides or {}
    result: dict[str, str] = {}
    for p in params:
        if p.location != ParamLocation.HEADER:
            continue
        if p.name in overrides:
            result[p.name] = str(overrides[p.name])
        elif p.required:
            result[p.name] = str(_smart_value(p.name, p.param_type))
    return result


# ---------------------------------------------------------------------------
# Per-category generators
# ---------------------------------------------------------------------------


def _gen_happy_path(
    endpoint: APIEndpoint,
    slug: str,
    counter: _Counter,
) -> list[TestCase]:
    """Generate happy-path test cases for *endpoint*."""
    cases: list[TestCase] = []
    _, url_params = _fill_path_params(endpoint.path, endpoint.parameters, method=endpoint.method)

    # --- 1. Basic request with required params ---
    body: Any = None
    if endpoint.method in _BODY_METHODS:
        body = _generate_body_from_schema(endpoint.request_body_schema)

    cases.append(TestCase(
        id=f"happy_path_{endpoint.method.value}_{slug}_{counter.next()}",
        name=f"Basic valid request to {endpoint.method.value} {endpoint.path}",
        description="Send a valid request with all required parameters and expect success.",
        category=TestCategory.HAPPY_PATH,
        severity=TestSeverity.MEDIUM,
        endpoint_path=endpoint.path,
        method=endpoint.method,
        url_params=url_params,
        query_params=_build_query_params(endpoint.parameters),
        headers=_build_headers(endpoint.parameters),
        body=body,
        expected_status=[200, 201, 204],
    ))

    # --- 2. Request with all optional params ---
    optional_query = [
        p for p in endpoint.parameters
        if p.location == ParamLocation.QUERY and not p.required
    ]
    if optional_query:
        cases.append(TestCase(
            id=f"happy_path_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"Request with all optional params to {endpoint.method.value} {endpoint.path}",
            description="Include every optional query parameter with valid values.",
            category=TestCategory.HAPPY_PATH,
            severity=TestSeverity.MEDIUM,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=_build_query_params(endpoint.parameters, include_optional=True),
            headers=_build_headers(endpoint.parameters),
            body=body,
            expected_status=[200, 201, 204],
        ))

    # --- 3. Enum parameter values ---
    for param in endpoint.parameters:
        enum_values = param.constraints.get("enum_values")
        if not enum_values or not isinstance(enum_values, list):
            continue
        for enum_val in enum_values:
            override = {param.name: enum_val}
            q = _build_query_params(endpoint.parameters, overrides=override)
            if param.location == ParamLocation.PATH:
                _, up = _fill_path_params(endpoint.path, endpoint.parameters, overrides=override, method=endpoint.method)
            else:
                up = url_params

            cases.append(TestCase(
                id=f"happy_path_{endpoint.method.value}_{slug}_{counter.next()}",
                name=f"Enum value '{enum_val}' for param '{param.name}' on {endpoint.path}",
                description=f"Test accepted enum value '{enum_val}' for parameter '{param.name}'.",
                category=TestCategory.HAPPY_PATH,
                severity=TestSeverity.MEDIUM,
                endpoint_path=endpoint.path,
                method=endpoint.method,
                url_params=up,
                query_params=q,
                headers=_build_headers(endpoint.parameters),
                body=body,
                expected_status=[200, 201, 204],
            ))

    return cases


def _gen_edge_cases(
    endpoint: APIEndpoint,
    slug: str,
    counter: _Counter,
) -> list[TestCase]:
    """Generate edge-case test cases for *endpoint*."""
    cases: list[TestCase] = []
    _, url_params = _fill_path_params(endpoint.path, endpoint.parameters, method=endpoint.method)
    base_query = _build_query_params(endpoint.parameters)
    base_headers = _build_headers(endpoint.parameters)

    body: Any = None
    if endpoint.method in _BODY_METHODS:
        body = _generate_body_from_schema(endpoint.request_body_schema)

    for param in endpoint.parameters:
        # Skip path params for most edge-case mutations -- they would produce
        # 404s rather than testing edge behaviour of the business logic.
        if param.location == ParamLocation.PATH:
            continue

        param_cases: list[tuple[str, Any]] = []

        # --- Empty / zero / boundary values by type ---
        if param.param_type == ParamType.STRING:
            param_cases.append(("empty string", ""))
            param_cases.append(("very long string (10000 chars)", "a" * 10_000))
            param_cases.append(("unicode characters", "éàüñ☃🚀"))

        elif param.param_type in (ParamType.INTEGER, ParamType.FLOAT):
            param_cases.append(("zero", 0))
            param_cases.append(("negative number", -1))

        elif param.param_type == ParamType.BOOLEAN:
            param_cases.append(("boolean as string 'true'", "true"))
            param_cases.append(("boolean as string 'false'", "false"))

        elif param.param_type == ParamType.ARRAY:
            param_cases.append(("empty array", []))

        elif param.param_type == ParamType.OBJECT:
            param_cases.append(("empty object", {}))

        # --- Boundary values from constraints ---
        minimum = param.constraints.get("minimum")
        maximum = param.constraints.get("maximum")
        min_length = param.constraints.get("min_length")
        max_length = param.constraints.get("max_length")

        if minimum is not None:
            param_cases.append((f"minimum boundary ({minimum})", minimum))
        if maximum is not None:
            param_cases.append((f"maximum boundary ({maximum})", maximum))
        if min_length is not None and param.param_type == ParamType.STRING:
            param_cases.append((f"min_length boundary ({min_length})", "a" * min_length))
        if max_length is not None and param.param_type == ParamType.STRING:
            param_cases.append((f"max_length boundary ({max_length})", "a" * max_length))

        for label, value in param_cases:
            override = {param.name: value}
            q = _build_query_params(endpoint.parameters, overrides=override)

            # If the param is in the body, mutate the body instead.
            test_body = body
            if param.location == ParamLocation.BODY and isinstance(body, dict):
                test_body = {**body, param.name: value}
            elif param.location == ParamLocation.QUERY:
                q = {**base_query, param.name: value}

            cases.append(TestCase(
                id=f"edge_case_{endpoint.method.value}_{slug}_{counter.next()}",
                name=f"Edge case: {label} for '{param.name}' on {endpoint.path}",
                description=f"Test edge-case value ({label}) for parameter '{param.name}'.",
                category=TestCategory.EDGE_CASE,
                severity=TestSeverity.MEDIUM,
                endpoint_path=endpoint.path,
                method=endpoint.method,
                url_params=url_params,
                query_params=q,
                headers=base_headers,
                body=test_body,
                expected_status=[200, 201, 204, 400, 422],
            ))

    # --- Missing optional params (should still work) ---
    optional_params = [
        p for p in endpoint.parameters
        if not p.required and p.location != ParamLocation.PATH
    ]
    if optional_params:
        # Build query with only required params (omit all optional).
        required_only_query = _build_query_params(endpoint.parameters, include_optional=False)
        minimal_body = _generate_empty_body_from_schema(endpoint.request_body_schema) if endpoint.method in _BODY_METHODS else None

        cases.append(TestCase(
            id=f"edge_case_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"Missing all optional params on {endpoint.path}",
            description="Omit every optional parameter; the request should still succeed.",
            category=TestCategory.EDGE_CASE,
            severity=TestSeverity.MEDIUM,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=required_only_query,
            headers=base_headers,
            body=minimal_body,
            expected_status=[200, 201, 204],
        ))

    return cases


def _gen_error_handling(
    endpoint: APIEndpoint,
    slug: str,
    counter: _Counter,
    sibling_methods: set[HTTPMethod] | None = None,
) -> list[TestCase]:
    """Generate error-handling test cases for *endpoint*."""
    cases: list[TestCase] = []
    _, url_params = _fill_path_params(endpoint.path, endpoint.parameters, method=endpoint.method)
    base_query = _build_query_params(endpoint.parameters)
    base_headers = _build_headers(endpoint.parameters)

    body: Any = None
    if endpoint.method in _BODY_METHODS:
        body = _generate_body_from_schema(endpoint.request_body_schema)

    # --- 1. Missing each required parameter ---
    required_params = [
        p for p in endpoint.parameters
        if p.required and p.location != ParamLocation.PATH
    ]
    for param in required_params:
        q = {k: v for k, v in base_query.items() if k != param.name}
        test_body = body
        if param.location == ParamLocation.BODY and isinstance(body, dict):
            test_body = {k: v for k, v in body.items() if k != param.name}

        cases.append(TestCase(
            id=f"error_handling_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"Missing required param '{param.name}' on {endpoint.path}",
            description=f"Omit the required parameter '{param.name}' and expect a validation error.",
            category=TestCategory.ERROR_HANDLING,
            severity=TestSeverity.HIGH,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=q,
            headers=base_headers,
            body=test_body,
            expected_status=[400, 422],
        ))

    # --- 2. Wrong type for each parameter ---
    # Skip STRING params: most values coerce to strings, so sending an
    # integer to a string field is not a meaningful type-error test.
    typed_params = [
        p for p in endpoint.parameters
        if p.location != ParamLocation.PATH
        and p.param_type not in (ParamType.ANY, ParamType.STRING)
    ]
    for param in typed_params:
        bad_value = _wrong_type_value(param.param_type)
        override = {param.name: bad_value}
        q = {**base_query, **{param.name: bad_value}} if param.location == ParamLocation.QUERY else base_query

        test_body = body
        if param.location == ParamLocation.BODY and isinstance(body, dict):
            test_body = {**body, param.name: bad_value}

        cases.append(TestCase(
            id=f"error_handling_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"Wrong type for '{param.name}' on {endpoint.path}",
            description=(
                f"Send a value of incorrect type for parameter '{param.name}' "
                f"(expected {param.param_type.value})."
            ),
            category=TestCategory.ERROR_HANDLING,
            severity=TestSeverity.HIGH,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=q,
            headers=base_headers,
            body=test_body,
            expected_status=[400, 422],
        ))

    # --- 3. Wrong HTTP method ---
    # Skip methods that are actually registered as sibling endpoints on
    # the same path (e.g. don't flag POST as wrong for GET /tasks when
    # POST /tasks is a real endpoint).
    known_methods = sibling_methods or {endpoint.method}
    for wrong_method in _ALL_METHODS:
        if wrong_method in known_methods:
            continue
        cases.append(TestCase(
            id=f"error_handling_{wrong_method.value}_{slug}_{counter.next()}",
            name=f"Wrong method {wrong_method.value} on {endpoint.path} (expects {endpoint.method.value})",
            description=(
                f"Send a {wrong_method.value} request to an endpoint that only "
                f"accepts {endpoint.method.value}."
            ),
            category=TestCategory.ERROR_HANDLING,
            severity=TestSeverity.HIGH,
            endpoint_path=endpoint.path,
            method=wrong_method,
            url_params=url_params,
            query_params=base_query if wrong_method == HTTPMethod.GET else {},
            headers=base_headers,
            body=None,
            expected_status=[405, 404, 400, 422],
        ))

    # --- 4. Extra unknown parameters ---
    extra_query = {**base_query, "__unknown_param__": "unexpected_value"}
    # DELETE endpoints may have already removed the resource during the happy-
    # path test, so accept 404 alongside the normal expected statuses.
    _extra_q_statuses = (
        [200, 201, 204, 400, 404, 422]
        if endpoint.method == HTTPMethod.DELETE
        else [200, 201, 204, 400, 422]
    )
    cases.append(TestCase(
        id=f"error_handling_{endpoint.method.value}_{slug}_{counter.next()}",
        name=f"Extra unknown query param on {endpoint.path}",
        description="Include an unrecognised query parameter; the API should ignore it or return an error.",
        category=TestCategory.ERROR_HANDLING,
        severity=TestSeverity.HIGH,
        endpoint_path=endpoint.path,
        method=endpoint.method,
        url_params=url_params,
        query_params=extra_query,
        headers=base_headers,
        body=body,
        expected_status=_extra_q_statuses,
    ))

    if endpoint.method in _BODY_METHODS and isinstance(body, dict):
        extra_body = {**body, "__unknown_field__": "unexpected_value"}
        cases.append(TestCase(
            id=f"error_handling_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"Extra unknown field in request body on {endpoint.path}",
            description="Include an unrecognised field in the body; the API should ignore it or return an error.",
            category=TestCategory.ERROR_HANDLING,
            severity=TestSeverity.HIGH,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=base_query,
            headers=base_headers,
            body=extra_body,
            expected_status=[200, 201, 204, 400, 422],
        ))

    return cases


# SQL injection payloads.
_SQL_INJECTION_PAYLOADS: list[str] = [
    "' OR '1'='1",
    "'; DROP TABLE users; --",
]

# XSS payloads.
_XSS_PAYLOADS: list[str] = [
    "<script>alert('xss')</script>",
]

# Path traversal payloads.
_PATH_TRAVERSAL_PAYLOADS: list[str] = [
    "../../etc/passwd",
]


def _gen_security(
    endpoint: APIEndpoint,
    slug: str,
    counter: _Counter,
) -> list[TestCase]:
    """Generate security-focused test cases for *endpoint*."""
    cases: list[TestCase] = []
    _, url_params = _fill_path_params(endpoint.path, endpoint.parameters, method=endpoint.method)
    base_query = _build_query_params(endpoint.parameters)
    base_headers = _build_headers(endpoint.parameters)

    body: Any = None
    if endpoint.method in _BODY_METHODS:
        body = _generate_body_from_schema(endpoint.request_body_schema)

    # Collect all string parameters (query + body) for injection testing.
    string_params: list[APIParameter] = [
        p for p in endpoint.parameters
        if p.param_type == ParamType.STRING and p.location != ParamLocation.PATH
    ]

    # Also include body-schema string fields as synthetic APIParameter objects
    # so that injection tests cover the request body.
    body_string_fields: list[str] = []
    if endpoint.request_body_schema:
        for field_name, field_def in endpoint.request_body_schema.get("properties", {}).items():
            if field_def.get("type", "string") == "string":
                body_string_fields.append(field_name)

    # --- SQL injection ---
    for payload in _SQL_INJECTION_PAYLOADS:
        for param in string_params:
            override = {param.name: payload}
            q = {**base_query, **override} if param.location == ParamLocation.QUERY else base_query

            test_body = body
            if param.location == ParamLocation.BODY and isinstance(body, dict):
                test_body = {**body, param.name: payload}

            cases.append(TestCase(
                id=f"security_{endpoint.method.value}_{slug}_{counter.next()}",
                name=f"SQL injection in '{param.name}' on {endpoint.path}",
                description=f"Inject SQL payload into '{param.name}': {payload!r}",
                category=TestCategory.SECURITY,
                severity=TestSeverity.CRITICAL,
                endpoint_path=endpoint.path,
                method=endpoint.method,
                url_params=url_params,
                query_params=q,
                headers=base_headers,
                body=test_body,
                expected_status=[200, 201, 204, 400, 422],
                expected_body_not_contains=["syntax error", "SQL", "mysql", "sqlite", "postgresql"],
            ))

        # Body-only string fields (not already in string_params).
        for field_name in body_string_fields:
            if any(p.name == field_name for p in string_params):
                continue
            if not isinstance(body, dict):
                continue

            cases.append(TestCase(
                id=f"security_{endpoint.method.value}_{slug}_{counter.next()}",
                name=f"SQL injection in body field '{field_name}' on {endpoint.path}",
                description=f"Inject SQL payload into body field '{field_name}': {payload!r}",
                category=TestCategory.SECURITY,
                severity=TestSeverity.CRITICAL,
                endpoint_path=endpoint.path,
                method=endpoint.method,
                url_params=url_params,
                query_params=base_query,
                headers=base_headers,
                body={**body, field_name: payload},
                expected_status=[200, 201, 204, 400, 422],
                expected_body_not_contains=["syntax error", "SQL", "mysql", "sqlite", "postgresql"],
            ))

    # --- XSS ---
    for payload in _XSS_PAYLOADS:
        for param in string_params:
            override = {param.name: payload}
            q = {**base_query, **override} if param.location == ParamLocation.QUERY else base_query

            test_body = body
            if param.location == ParamLocation.BODY and isinstance(body, dict):
                test_body = {**body, param.name: payload}

            cases.append(TestCase(
                id=f"security_{endpoint.method.value}_{slug}_{counter.next()}",
                name=f"XSS in '{param.name}' on {endpoint.path}",
                description=f"Inject XSS payload into '{param.name}': {payload!r}",
                category=TestCategory.SECURITY,
                severity=TestSeverity.CRITICAL,
                endpoint_path=endpoint.path,
                method=endpoint.method,
                url_params=url_params,
                query_params=q,
                headers=base_headers,
                body=test_body,
                expected_status=[200, 201, 204, 400, 422],
                expected_body_not_contains=["<script>"],
            ))

    # --- Path traversal ---
    for payload in _PATH_TRAVERSAL_PAYLOADS:
        for param in string_params:
            override = {param.name: payload}
            q = {**base_query, **override} if param.location == ParamLocation.QUERY else base_query

            test_body = body
            if param.location == ParamLocation.BODY and isinstance(body, dict):
                test_body = {**body, param.name: payload}

            cases.append(TestCase(
                id=f"security_{endpoint.method.value}_{slug}_{counter.next()}",
                name=f"Path traversal in '{param.name}' on {endpoint.path}",
                description=f"Inject path-traversal payload into '{param.name}': {payload!r}",
                category=TestCategory.SECURITY,
                severity=TestSeverity.CRITICAL,
                endpoint_path=endpoint.path,
                method=endpoint.method,
                url_params=url_params,
                query_params=q,
                headers=base_headers,
                body=test_body,
                expected_status=[200, 201, 204, 400, 422],
                expected_body_not_contains=["root:", "/etc/passwd"],
            ))

    # --- Null bytes ---
    null_payload = "test\x00value"
    for param in string_params:
        override = {param.name: null_payload}
        q = {**base_query, **override} if param.location == ParamLocation.QUERY else base_query

        test_body = body
        if param.location == ParamLocation.BODY and isinstance(body, dict):
            test_body = {**body, param.name: null_payload}

        cases.append(TestCase(
            id=f"security_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"Null byte in '{param.name}' on {endpoint.path}",
            description=f"Inject null byte into '{param.name}' to test binary safety.",
            category=TestCategory.SECURITY,
            severity=TestSeverity.CRITICAL,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=q,
            headers=base_headers,
            body=test_body,
            expected_status=[200, 201, 204, 400, 422],
        ))

    # --- Auth required but no credentials ---
    if endpoint.requires_auth:
        cases.append(TestCase(
            id=f"security_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"No auth credentials on {endpoint.path}",
            description="Call an endpoint that requires authentication without providing any credentials.",
            category=TestCategory.SECURITY,
            severity=TestSeverity.CRITICAL,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=base_query,
            headers={},  # deliberately empty -- strip auth headers
            body=body,
            expected_status=[401, 403],
        ))

    # --- Very large payload ---
    if endpoint.method in _BODY_METHODS:
        large_body = {"data": "x" * 1_000_000}
        cases.append(TestCase(
            id=f"security_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"Very large payload (1 MB) on {endpoint.path}",
            description="Send a ~1 MB request body to verify the server does not crash or hang.",
            category=TestCategory.SECURITY,
            severity=TestSeverity.CRITICAL,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=base_query,
            headers=base_headers,
            body=large_body,
            expected_status=[200, 201, 204, 400, 413, 422],
            timeout_seconds=30.0,
        ))

    return cases


def _gen_schema_validation(
    endpoint: APIEndpoint,
    slug: str,
    counter: _Counter,
) -> list[TestCase]:
    """Generate schema-validation test cases for *endpoint*."""
    cases: list[TestCase] = []
    _, url_params = _fill_path_params(endpoint.path, endpoint.parameters, method=endpoint.method)
    base_query = _build_query_params(endpoint.parameters)
    base_headers = _build_headers(endpoint.parameters)

    body: Any = None
    if endpoint.method in _BODY_METHODS:
        body = _generate_body_from_schema(endpoint.request_body_schema)

    # For DELETE endpoints, the happy-path test may have already removed the
    # resource, so all schema-validation probes should tolerate 404 as well.
    expected_success: list[int] = [200, 201, 204]
    if endpoint.method == HTTPMethod.DELETE:
        expected_success = [200, 201, 204, 404]

    # --- Response should be valid JSON ---
    cases.append(TestCase(
        id=f"schema_validation_{endpoint.method.value}_{slug}_{counter.next()}",
        name=f"Response is valid JSON from {endpoint.method.value} {endpoint.path}",
        description="Verify the response body is parseable as valid JSON.",
        category=TestCategory.SCHEMA_VALIDATION,
        severity=TestSeverity.HIGH,
        endpoint_path=endpoint.path,
        method=endpoint.method,
        url_params=url_params,
        query_params=base_query,
        headers=base_headers,
        body=body,
        expected_status=expected_success,
    ))

    # --- Content-Type header validation ---
    expected_content_types: list[str] = []
    for resp in endpoint.responses:
        if resp.content_type and resp.content_type not in expected_content_types:
            expected_content_types.append(resp.content_type)

    if expected_content_types:
        cases.append(TestCase(
            id=f"schema_validation_{endpoint.method.value}_{slug}_{counter.next()}",
            name=f"Content-Type matches expected on {endpoint.method.value} {endpoint.path}",
            description=(
                f"Verify the Content-Type header is one of: "
                f"{', '.join(expected_content_types)}."
            ),
            category=TestCategory.SCHEMA_VALIDATION,
            severity=TestSeverity.HIGH,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=base_query,
            headers=base_headers,
            body=body,
            expected_status=expected_success,
        ))

    # --- Response schema validation (when a schema is defined) ---
    for resp in endpoint.responses:
        if not resp.schema_def:
            continue
        cases.append(TestCase(
            id=f"schema_validation_{endpoint.method.value}_{slug}_{counter.next()}",
            name=(
                f"Response matches schema for status {resp.status_code} "
                f"on {endpoint.method.value} {endpoint.path}"
            ),
            description=(
                f"Validate that the {resp.status_code} response conforms to "
                f"the declared schema."
            ),
            category=TestCategory.SCHEMA_VALIDATION,
            severity=TestSeverity.HIGH,
            endpoint_path=endpoint.path,
            method=endpoint.method,
            url_params=url_params,
            query_params=base_query,
            headers=base_headers,
            body=body,
            expected_status=expected_success,
            expected_schema=resp.schema_def,
        ))

    return cases


# ---------------------------------------------------------------------------
# Counter utility (deterministic, not random)
# ---------------------------------------------------------------------------


class _Counter:
    """Simple auto-incrementing counter for generating unique test-case IDs."""

    __slots__ = ("_value",)

    def __init__(self, start: int = 0) -> None:
        self._value = start

    def next(self) -> int:
        """Return the current value and advance by one."""
        val = self._value
        self._value += 1
        return val


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

# Callable mapping: adding a new category only requires writing the generator
# function and adding one entry here.
_GENERATORS: dict[
    TestCategory,
    Any,  # Callable[[APIEndpoint, str, _Counter], list[TestCase]]
] = {
    TestCategory.HAPPY_PATH: _gen_happy_path,
    TestCategory.EDGE_CASE: _gen_edge_cases,
    TestCategory.ERROR_HANDLING: _gen_error_handling,
    TestCategory.SECURITY: _gen_security,
    TestCategory.SCHEMA_VALIDATION: _gen_schema_validation,
}

# Default categories when the caller does not specify a filter.
_DEFAULT_CATEGORIES: list[TestCategory] = [
    TestCategory.HAPPY_PATH,
    TestCategory.EDGE_CASE,
    TestCategory.ERROR_HANDLING,
    TestCategory.SECURITY,
    TestCategory.SCHEMA_VALIDATION,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class TestGenerator:
    """Automatically generate comprehensive test suites from API specifications.

    The generator examines each endpoint in an :class:`APISpec` and produces
    :class:`TestCase` instances covering happy paths, edge cases, error
    handling, security probes, and schema validation.  All generated values
    are deterministic.

    Parameters
    ----------
    categories:
        Optional list of :class:`TestCategory` values to limit generation.
        When *None*, all supported categories are produced.
    """

    def __init__(self, categories: list[TestCategory] | None = None) -> None:
        self._categories: list[TestCategory] = categories or list(_DEFAULT_CATEGORIES)
        logger.debug(
            "TestGenerator initialised with categories: %s",
            [c.value for c in self._categories],
        )

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def generate(self, spec: APISpec) -> TestSuite:
        """Generate a complete test suite from an API specification.

        Iterates over every endpoint in *spec* and delegates to
        :meth:`generate_for_endpoint` for the actual test-case creation.

        Parameters
        ----------
        spec:
            A fully populated :class:`APISpec` (typically produced by the
            discovery engine).

        Returns
        -------
        TestSuite
            A suite containing all generated test cases (results are empty
            until a runner executes them).
        """
        logger.info(
            "Generating test suite for '%s' (%d endpoint(s), categories=%s)",
            spec.name,
            len(spec.endpoints),
            [c.value for c in self._categories],
        )

        # Build a map of path → set of methods for sibling-aware generation.
        # This lets the "wrong method" generator skip methods that are actually
        # valid on the same path (e.g. GET /tasks and POST /tasks).
        self._path_methods: dict[str, set[HTTPMethod]] = {}
        for ep in spec.endpoints:
            self._path_methods.setdefault(ep.path, set()).add(ep.method)

        all_cases: list[TestCase] = []
        for endpoint in spec.endpoints:
            cases = self.generate_for_endpoint(endpoint)
            all_cases.extend(cases)
            logger.debug(
                "  %s %s -> %d test case(s)",
                endpoint.method.value,
                endpoint.path,
                len(cases),
            )

        suite = TestSuite(
            api_name=spec.name,
            api_source=spec.source_path,
            test_cases=all_cases,
        )

        logger.info(
            "Generated %d test case(s) across %d endpoint(s) for '%s'",
            len(all_cases),
            len(spec.endpoints),
            spec.name,
        )
        return suite

    def generate_for_endpoint(self, endpoint: APIEndpoint) -> list[TestCase]:
        """Generate all test cases for a single endpoint.

        Parameters
        ----------
        endpoint:
            The endpoint to test.

        Returns
        -------
        list[TestCase]
            Ordered list of test cases spanning the configured categories.
        """
        slug = _slugify_path(endpoint.path)
        counter = _Counter()
        cases: list[TestCase] = []

        # Gather sibling methods for this path (other methods registered
        # on the same URL path).  Used by the error-handling generator to
        # avoid flagging valid methods as "wrong".
        sibling_methods = self._path_methods.get(endpoint.path, {endpoint.method})

        for category in self._categories:
            gen_fn = _GENERATORS.get(category)
            if gen_fn is None:
                logger.warning("No generator registered for category %s", category.value)
                continue

            # Pass sibling_methods to error_handling generator
            if category == TestCategory.ERROR_HANDLING:
                category_cases = gen_fn(endpoint, slug, counter, sibling_methods=sibling_methods)
            else:
                category_cases = gen_fn(endpoint, slug, counter)
            cases.extend(category_cases)

        return cases
