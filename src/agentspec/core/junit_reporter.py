"""JUnit XML report generator for CI system integration.

Produces standard JUnit XML output compatible with Jenkins, GitHub Actions,
GitLab CI, Azure DevOps, and other CI/CD systems that consume JUnit reports.

Test results are grouped by :class:`~agentspec.models.test.TestCategory` into
separate ``<testsuite>`` elements within a root ``<testsuites>`` wrapper.

Usage::

    from agentspec.core.junit_reporter import JUnitReporter

    reporter = JUnitReporter()
    xml_string = reporter.generate(suite)
    reporter.save(suite, "agentspec-results.xml")
"""

from __future__ import annotations

import logging
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

from agentspec.models.test import (
    TestCategory,
    TestResult,
    TestStatus,
    TestSuite,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_XML_DECLARATION: str = '<?xml version="1.0" encoding="UTF-8"?>\n'

# Deterministic ordering for categories in the output.
_CATEGORY_ORDER: list[TestCategory] = [
    TestCategory.HAPPY_PATH,
    TestCategory.EDGE_CASE,
    TestCategory.ERROR_HANDLING,
    TestCategory.SECURITY,
    TestCategory.SCHEMA_VALIDATION,
    TestCategory.PERFORMANCE,
]

_CATEGORY_LABELS: dict[TestCategory, str] = {
    TestCategory.HAPPY_PATH: "happy_path",
    TestCategory.EDGE_CASE: "edge_case",
    TestCategory.ERROR_HANDLING: "error_handling",
    TestCategory.SECURITY: "security",
    TestCategory.SCHEMA_VALIDATION: "schema_validation",
    TestCategory.PERFORMANCE: "performance",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ms_to_seconds(duration_ms: float) -> str:
    """Convert milliseconds to seconds, formatted to three decimal places."""
    return f"{duration_ms / 1000.0:.3f}"


def _count_by_status(
    results: list[TestResult],
    status: TestStatus,
) -> int:
    """Count results matching a given status."""
    return sum(1 for r in results if r.status == status)


def _build_failure_body(result: TestResult) -> str:
    """Build a human-readable failure body from assertion failures.

    Each assertion failure is rendered on its own line with expected/actual
    values.
    """
    lines: list[str] = []
    for failure in result.failures:
        lines.append(f"Assertion: {failure.assertion}")
        lines.append(f"  Expected: {failure.expected}")
        lines.append(f"  Actual: {failure.actual}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _build_failure_message(result: TestResult) -> str:
    """Build a concise failure message suitable for the ``message`` attribute."""
    if result.failures:
        first = result.failures[0]
        return f"{first.assertion}: expected {first.expected}, got {first.actual}"
    return "assertion failed"


# ---------------------------------------------------------------------------
# XML tree builders
# ---------------------------------------------------------------------------


def _build_testcase_element(
    result: TestResult,
    parent: Element,
) -> Element:
    """Create a ``<testcase>`` element for a single test result.

    Parameters
    ----------
    result:
        The test result to represent.
    parent:
        The parent ``<testsuite>`` element.

    Returns
    -------
    Element
        The new ``<testcase>`` element (already appended to *parent*).
    """
    classname = f"agentspec.{result.category.value}"

    testcase = SubElement(
        parent,
        "testcase",
        name=result.test_name,
        classname=classname,
        time=_ms_to_seconds(result.duration_ms),
    )

    if result.status == TestStatus.FAILED:
        failure_msg = _build_failure_message(result)
        failure_body = _build_failure_body(result)

        failure_elem = SubElement(
            testcase,
            "failure",
            message=failure_msg,
            type="AssertionFailure",
        )
        failure_elem.text = failure_body if failure_body else None

    elif result.status == TestStatus.ERROR:
        error_elem = SubElement(testcase, "system-err")
        error_elem.text = result.error_message or "unknown error"

    elif result.status == TestStatus.SKIPPED:
        SubElement(
            testcase,
            "skipped",
            message=result.error_message or "skipped",
        )

    return testcase


def _build_testsuite_element(
    category: TestCategory,
    results: list[TestResult],
    parent: Element,
) -> Element:
    """Create a ``<testsuite>`` element grouping results for one category.

    Parameters
    ----------
    category:
        The test category this suite represents.
    results:
        All results belonging to this category.
    parent:
        The root ``<testsuites>`` element.

    Returns
    -------
    Element
        The new ``<testsuite>`` element (already appended to *parent*).
    """
    suite_name = _CATEGORY_LABELS.get(category, category.value)
    total_duration_ms = sum(r.duration_ms for r in results)

    suite_elem = SubElement(
        parent,
        "testsuite",
        name=suite_name,
        tests=str(len(results)),
        failures=str(_count_by_status(results, TestStatus.FAILED)),
        errors=str(_count_by_status(results, TestStatus.ERROR)),
        skipped=str(_count_by_status(results, TestStatus.SKIPPED)),
        time=_ms_to_seconds(total_duration_ms),
    )

    for result in results:
        _build_testcase_element(result, suite_elem)

    return suite_elem


def _build_testsuites_element(suite: TestSuite) -> Element:
    """Build the root ``<testsuites>`` element from a complete test suite.

    Results are grouped by category into separate ``<testsuite>`` children.
    Categories with no results are omitted.

    Parameters
    ----------
    suite:
        The completed test suite.

    Returns
    -------
    Element
        The root XML element.
    """
    root = Element(
        "testsuites",
        name="agentspec",
        tests=str(suite.total),
        failures=str(suite.failed),
        errors=str(suite.errors),
        skipped=str(suite.skipped),
        time=_ms_to_seconds(suite.duration_ms),
    )

    by_category = suite.by_category()

    # Use deterministic ordering; append any unexpected categories at the end.
    ordered_categories = list(_CATEGORY_ORDER)
    for cat in by_category:
        if cat not in ordered_categories:
            ordered_categories.append(cat)

    for category in ordered_categories:
        results = by_category.get(category)
        if not results:
            continue
        _build_testsuite_element(category, results, root)

    return root


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class JUnitReporter:
    """Generate JUnit XML reports from agentspec test suites.

    The output conforms to the JUnit XML schema understood by major CI
    systems.  Test cases are grouped by
    :class:`~agentspec.models.test.TestCategory` into ``<testsuite>``
    elements.

    Example
    -------
    ::

        reporter = JUnitReporter()
        xml = reporter.generate(suite)
        print(xml)

        path = reporter.save(suite, "results.xml")
        print(f"Report written to {path}")
    """

    def __init__(self) -> None:
        pass

    def generate(self, suite: TestSuite) -> str:
        """Generate a JUnit XML string from test results.

        Parameters
        ----------
        suite:
            The completed test suite (with results populated).

        Returns
        -------
        str
            A complete JUnit XML document as a string, including the XML
            declaration.
        """
        root = _build_testsuites_element(suite)

        # ``tostring`` with xml_declaration is only available in Python 3.8+
        # and adds a standalone attribute we don't want.  We prepend the
        # declaration manually for a cleaner output.
        xml_body = tostring(root, encoding="unicode", xml_declaration=False)
        xml_output = _XML_DECLARATION + xml_body

        logger.debug(
            "Generated JUnit XML: %d tests, %d failures, %d errors",
            suite.total,
            suite.failed,
            suite.errors,
        )

        return xml_output

    def save(self, suite: TestSuite, output_path: str) -> str:
        """Save JUnit XML to a file.

        Creates parent directories if they do not exist.

        Parameters
        ----------
        suite:
            The completed test suite (with results populated).
        output_path:
            Filesystem path where the XML file will be written.

        Returns
        -------
        str
            The absolute path to the written file.
        """
        xml_content = self.generate(suite)

        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(xml_content, encoding="utf-8")

        logger.info("JUnit XML report saved to %s", out)
        return str(out)
