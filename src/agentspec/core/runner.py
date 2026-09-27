"""Async test runner --- execute test cases against a live API server.

The :class:`TestRunner` takes a :class:`TestSuite` (populated with
:class:`TestCase` instances by the generator) and fires real HTTP requests at
the target API, collecting timing data, status codes, and response bodies.
Assertions are evaluated against each response to produce a :class:`TestResult`.

Concurrency is managed with :mod:`anyio` task groups and a capacity limiter so
that the target server is not overwhelmed.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable

import anyio
import httpx

from agentspec.models.test import (
    AssertionFailure,
    TestCase,
    TestResult,
    TestStatus,
    TestSuite,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path parameter substitution
# ---------------------------------------------------------------------------

_PATH_PARAM_RE: re.Pattern[str] = re.compile(r"\{(\w+)\}")


def _build_url(base_url: str, endpoint_path: str, url_params: dict[str, Any]) -> str:
    """Construct the full request URL.

    Substitutes ``{param}`` placeholders in *endpoint_path* with values from
    *url_params*, then joins the result with *base_url*.

    Parameters
    ----------
    base_url:
        The root URL of the API server (e.g. ``http://localhost:8000``).
    endpoint_path:
        The path template (e.g. ``/users/{user_id}/posts``).
    url_params:
        Mapping of parameter names to substitution values.

    Returns
    -------
    str
        The fully resolved URL.
    """
    # Substitute path parameters.
    resolved_path = _PATH_PARAM_RE.sub(
        lambda m: str(url_params.get(m.group(1), m.group(0))),
        endpoint_path,
    )

    # Normalise: strip trailing slash from base, ensure leading slash on path.
    base = base_url.rstrip("/")
    if not resolved_path.startswith("/"):
        resolved_path = "/" + resolved_path

    return base + resolved_path


# ---------------------------------------------------------------------------
# Schema validation (basic)
# ---------------------------------------------------------------------------


def _validate_schema(data: Any, schema: dict[str, Any]) -> list[AssertionFailure]:
    """Perform basic JSON-Schema-style validation on *data*.

    This is intentionally lightweight --- it checks ``type`` and ``required``
    keys without pulling in a full JSON Schema library.

    Parameters
    ----------
    data:
        The parsed response JSON.
    schema:
        A dict with optional ``type``, ``properties``, and ``required`` keys.

    Returns
    -------
    list[AssertionFailure]
        A list of failures (empty if validation passes).
    """
    failures: list[AssertionFailure] = []

    expected_type = schema.get("type")
    if expected_type:
        type_map: dict[str, tuple[type, ...]] = {
            "object": (dict,),
            "array": (list,),
            "string": (str,),
            "integer": (int,),
            "number": (int, float),
            "boolean": (bool,),
            "null": (type(None),),
        }
        allowed = type_map.get(expected_type)
        if allowed and not isinstance(data, allowed):
            actual_type = type(data).__name__
            failures.append(AssertionFailure(
                assertion="schema_type",
                expected=f"type '{expected_type}'",
                actual=f"type '{actual_type}'",
            ))
            # If the top-level type is wrong, skip deeper checks.
            return failures

    # Check required keys when data is a dict and schema declares properties.
    if isinstance(data, dict):
        required_keys: list[str] = schema.get("required", [])
        for key in required_keys:
            if key not in data:
                failures.append(AssertionFailure(
                    assertion="schema_required_key",
                    expected=f"key '{key}' present",
                    actual="key missing",
                ))

        # Validate nested property types when declared.
        properties: dict[str, Any] = schema.get("properties", {})
        for prop_name, prop_schema in properties.items():
            if prop_name not in data:
                continue  # Missing keys are caught by the required check.
            prop_type = prop_schema.get("type")
            if not prop_type:
                continue
            prop_type_map: dict[str, tuple[type, ...]] = {
                "string": (str,),
                "integer": (int,),
                "number": (int, float),
                "boolean": (bool,),
                "array": (list,),
                "object": (dict,),
                "null": (type(None),),
            }
            prop_allowed = prop_type_map.get(prop_type)
            if prop_allowed and not isinstance(data[prop_name], prop_allowed):
                failures.append(AssertionFailure(
                    assertion=f"schema_property_type[{prop_name}]",
                    expected=f"type '{prop_type}'",
                    actual=f"type '{type(data[prop_name]).__name__}'",
                ))

    return failures


# ---------------------------------------------------------------------------
# Assertion runner
# ---------------------------------------------------------------------------


def _run_assertions(test: TestCase, response: httpx.Response) -> list[AssertionFailure]:
    """Evaluate all assertions for a single test case against its response.

    Parameters
    ----------
    test:
        The test case whose expectations to check.
    response:
        The HTTP response received from the server.

    Returns
    -------
    list[AssertionFailure]
        Every assertion that did not hold (empty when all pass).
    """
    failures: list[AssertionFailure] = []

    # 1. Status code -----------------------------------------------------------
    if response.status_code not in test.expected_statuses:
        failures.append(AssertionFailure(
            assertion="status_code",
            expected=str(test.expected_statuses),
            actual=str(response.status_code),
        ))

    # 2. Body contains ---------------------------------------------------------
    response_text = response.text
    for needle in test.expected_body_contains:
        if needle not in response_text:
            failures.append(AssertionFailure(
                assertion="body_contains",
                expected=f"response contains {needle!r}",
                actual=f"not found in response ({len(response_text)} chars)",
            ))

    # 3. Body not contains -----------------------------------------------------
    for needle in test.expected_body_not_contains:
        if needle in response_text:
            failures.append(AssertionFailure(
                assertion="body_not_contains",
                expected=f"response does NOT contain {needle!r}",
                actual=f"found {needle!r} in response",
            ))

    # 4. Schema validation -----------------------------------------------------
    if test.expected_schema is not None:
        try:
            response_json = response.json()
        except (json.JSONDecodeError, ValueError):
            failures.append(AssertionFailure(
                assertion="schema_parse",
                expected="valid JSON response",
                actual="response is not valid JSON",
            ))
        else:
            schema_failures = _validate_schema(response_json, test.expected_schema)
            failures.extend(schema_failures)

    return failures


# ---------------------------------------------------------------------------
# TestRunner
# ---------------------------------------------------------------------------


class TestRunner:
    """Execute test cases against a live API server.

    The runner sends real HTTP requests using :mod:`httpx`, evaluates
    assertions on each response, and populates a :class:`TestSuite` with
    :class:`TestResult` objects.

    Parameters
    ----------
    base_url:
        Root URL of the API under test (e.g. ``http://localhost:8000``).
    concurrency:
        Maximum number of requests in flight at once.
    timeout:
        Default request timeout in seconds (overridden per-test by
        :attr:`TestCase.timeout_seconds`).
    """

    def __init__(
        self,
        base_url: str,
        concurrency: int = 10,
        timeout: float = 30.0,
    ) -> None:
        self._base_url = base_url
        self._concurrency = concurrency
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        suite: TestSuite,
        on_result: Callable[[TestResult], None] | None = None,
    ) -> TestSuite:
        """Run every test case in *suite* and return the suite with results.

        Tests are executed concurrently (up to :attr:`concurrency` at a time).
        The optional *on_result* callback is invoked after each test completes,
        which is useful for streaming progress to the console.

        Parameters
        ----------
        suite:
            A test suite whose :attr:`~TestSuite.test_cases` are populated.
        on_result:
            Optional callback fired with each :class:`TestResult` as it
            becomes available.

        Returns
        -------
        TestSuite
            The same *suite* instance, now with :attr:`~TestSuite.results`
            populated and timestamps set.
        """
        suite.started_at = datetime.now(timezone.utc)
        suite.results = []

        logger.info(
            "Starting test run: %d test(s) against %s (concurrency=%d)",
            len(suite.test_cases),
            self._base_url,
            self._concurrency,
        )

        limiter = anyio.CapacityLimiter(self._concurrency)

        async def _execute(test: TestCase) -> None:
            async with limiter:
                result = await self.run_single(test)
                suite.results.append(result)
                if on_result is not None:
                    on_result(result)

        async with anyio.create_task_group() as tg:
            for test in suite.test_cases:
                tg.start_soon(_execute, test)

        suite.completed_at = datetime.now(timezone.utc)

        logger.info(
            "Test run complete: %d passed, %d failed, %d errors, %d skipped "
            "(%.1f ms total)",
            suite.passed,
            suite.failed,
            suite.errors,
            suite.skipped,
            suite.duration_ms,
        )

        return suite

    async def run_single(self, test: TestCase) -> TestResult:
        """Execute a single test case and return its result.

        Builds the request from the test case fields, sends it via
        :mod:`httpx`, measures elapsed time, and runs all configured
        assertions.

        Parameters
        ----------
        test:
            The test case to execute.

        Returns
        -------
        TestResult
            The result including status, timing, and any assertion failures.
        """
        url = _build_url(self._base_url, test.endpoint_path, test.url_params)
        timeout = test.timeout_seconds or self._timeout

        logger.debug("Running test %s: %s %s", test.id, test.method.value, url)

        start = time.perf_counter()

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.request(
                    method=test.method.value,
                    url=url,
                    params=test.query_params or None,
                    headers=test.headers or None,
                    json=test.body if test.body is not None else None,
                )
        except httpx.TimeoutException as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("Test %s timed out after %.1f ms: %s", test.id, elapsed_ms, exc)
            return TestResult(
                test_id=test.id,
                test_name=test.name,
                category=test.category,
                severity=test.severity,
                status=TestStatus.ERROR,
                duration_ms=elapsed_ms,
                error_message=f"Request timed out after {timeout}s: {exc}",
            )
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("Test %s HTTP error after %.1f ms: %s", test.id, elapsed_ms, exc)
            return TestResult(
                test_id=test.id,
                test_name=test.name,
                category=test.category,
                severity=test.severity,
                status=TestStatus.ERROR,
                duration_ms=elapsed_ms,
                error_message=f"HTTP error: {exc}",
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Test %s unexpected error: %s", test.id, exc, exc_info=True)
            return TestResult(
                test_id=test.id,
                test_name=test.name,
                category=test.category,
                severity=test.severity,
                status=TestStatus.ERROR,
                duration_ms=elapsed_ms,
                error_message=f"Unexpected error: {exc}",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse response body for the result record.
        try:
            response_body = response.json()
        except (json.JSONDecodeError, ValueError):
            response_body = response.text

        # Run assertions.
        failures = _run_assertions(test, response)

        status = TestStatus.PASSED if not failures else TestStatus.FAILED

        logger.debug(
            "Test %s %s (status_code=%d, %.1f ms, %d failure(s))",
            test.id,
            status.value,
            response.status_code,
            elapsed_ms,
            len(failures),
        )

        return TestResult(
            test_id=test.id,
            test_name=test.name,
            category=test.category,
            severity=test.severity,
            status=status,
            duration_ms=elapsed_ms,
            status_code=response.status_code,
            response_body=response_body,
            failures=failures,
        )
