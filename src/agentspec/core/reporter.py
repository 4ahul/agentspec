"""Test result reporter --- format and display test suite outcomes.

Provides two output modes:

* **text** --- a rich, colourful terminal report using :mod:`rich` tables,
  panels, and progress bars.
* **json** --- a machine-readable JSON serialisation of the suite and a
  summary object, suitable for CI pipelines and downstream tooling.

The :class:`Reporter` also supports live progress output via
:meth:`print_progress`, which can be wired to the runner's ``on_result``
callback for real-time feedback.
"""

from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentspec.models.test import (
    TestCategory,
    TestResult,
    TestSeverity,
    TestStatus,
    TestSuite,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STATUS_ICONS: dict[TestStatus, str] = {
    TestStatus.PASSED: "✓",   # checkmark
    TestStatus.FAILED: "✗",   # ballot X
    TestStatus.ERROR: "⚠",    # warning sign
    TestStatus.SKIPPED: "○",  # white circle
}

_STATUS_COLOURS: dict[TestStatus, str] = {
    TestStatus.PASSED: "green",
    TestStatus.FAILED: "red",
    TestStatus.ERROR: "yellow",
    TestStatus.SKIPPED: "dim",
}

_SEVERITY_COLOURS: dict[TestSeverity, str] = {
    TestSeverity.CRITICAL: "bold red",
    TestSeverity.HIGH: "red",
    TestSeverity.MEDIUM: "yellow",
    TestSeverity.LOW: "cyan",
    TestSeverity.INFO: "dim",
}

_CATEGORY_LABELS: dict[TestCategory, str] = {
    TestCategory.HAPPY_PATH: "Happy Path",
    TestCategory.EDGE_CASE: "Edge Case",
    TestCategory.ERROR_HANDLING: "Error Handling",
    TestCategory.SECURITY: "Security",
    TestCategory.SCHEMA_VALIDATION: "Schema Validation",
    TestCategory.PERFORMANCE: "Performance",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _progress_bar(ratio: float, width: int = 30) -> Text:
    """Render a coloured progress bar as a :class:`rich.text.Text` object.

    Parameters
    ----------
    ratio:
        A value between 0.0 and 1.0 representing the fill proportion.
    width:
        Total character width of the bar.
    """
    filled = int(ratio * width)
    empty = width - filled

    if ratio >= 0.9:
        colour = "green"
    elif ratio >= 0.7:
        colour = "yellow"
    else:
        colour = "red"

    bar = Text()
    bar.append("█" * filled, style=colour)
    bar.append("░" * empty, style="dim")
    bar.append(f" {ratio * 100:.1f}%", style="bold " + colour)
    return bar


def _truncate(text: str, max_length: int = 80) -> str:
    """Truncate *text* to *max_length*, appending an ellipsis if trimmed."""
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _build_summary_dict(suite: TestSuite) -> dict[str, Any]:
    """Build a summary dict suitable for JSON serialisation."""
    return {
        "api_name": suite.api_name,
        "api_source": suite.api_source,
        "total": suite.total,
        "passed": suite.passed,
        "failed": suite.failed,
        "errors": suite.errors,
        "skipped": suite.skipped,
        "pass_rate": round(suite.pass_rate, 2),
        "duration_ms": round(suite.duration_ms, 2),
        "started_at": suite.started_at.isoformat() if suite.started_at else None,
        "completed_at": suite.completed_at.isoformat() if suite.completed_at else None,
        "critical_failures": len(suite.critical_failures),
    }


# ---------------------------------------------------------------------------
# Text report builder
# ---------------------------------------------------------------------------


def _build_text_report(suite: TestSuite, console: Console) -> None:
    """Render the full text report to *console*.

    Parameters
    ----------
    suite:
        The completed test suite.
    console:
        The Rich console to write to.
    """
    # --- Header panel ---------------------------------------------------------
    header_lines: list[str] = []
    header_lines.append(f"[bold]{suite.api_name}[/bold]")
    if suite.api_source:
        header_lines.append(f"Source: [dim]{suite.api_source}[/dim]")
    header_lines.append(f"Tests: [bold]{len(suite.test_cases)}[/bold] generated")

    console.print(Panel(
        "\n".join(header_lines),
        title="[bold blue]agentspec[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    ))

    if not suite.results:
        console.print("[dim]No results to display.[/dim]")
        return

    # --- Summary line ---------------------------------------------------------
    summary = Text()
    summary.append(f"{suite.passed} passed", style="bold green")
    summary.append(" | ")
    summary.append(f"{suite.failed} failed", style="bold red")
    summary.append(" | ")
    summary.append(f"{suite.errors} errors", style="bold yellow")
    if suite.skipped:
        summary.append(" | ")
        summary.append(f"{suite.skipped} skipped", style="dim")
    summary.append(f"  ({suite.duration_ms:.0f} ms)")

    console.print()
    console.print(summary)

    # --- Pass rate bar --------------------------------------------------------
    rate = suite.pass_rate / 100.0 if suite.total else 0.0
    console.print(_progress_bar(rate))
    console.print()

    # --- Results table grouped by category ------------------------------------
    by_cat = suite.by_category()

    # Use a deterministic category order.
    ordered_categories = [
        TestCategory.HAPPY_PATH,
        TestCategory.EDGE_CASE,
        TestCategory.ERROR_HANDLING,
        TestCategory.SECURITY,
        TestCategory.SCHEMA_VALIDATION,
        TestCategory.PERFORMANCE,
    ]

    for category in ordered_categories:
        results = by_cat.get(category)
        if not results:
            continue

        cat_label = _CATEGORY_LABELS.get(category, category.value)

        cat_passed = sum(1 for r in results if r.status == TestStatus.PASSED)
        cat_total = len(results)
        cat_summary = Text()
        cat_summary.append(f" {cat_label} ", style="bold")
        cat_summary.append(f"({cat_passed}/{cat_total} passed)", style="dim")

        console.print(cat_summary)

        table = Table(
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=True,
            pad_edge=False,
            show_edge=False,
        )
        table.add_column("", width=3, justify="center")      # status icon
        table.add_column("Test", ratio=4, no_wrap=False)
        table.add_column("Status", width=10)
        table.add_column("Code", width=6, justify="right")
        table.add_column("Time", width=10, justify="right")
        table.add_column("Detail", ratio=3, no_wrap=False)

        for result in results:
            icon = _STATUS_ICONS.get(result.status, "?")
            colour = _STATUS_COLOURS.get(result.status, "white")

            icon_text = Text(icon, style=colour)
            name_text = Text(result.test_name)
            status_text = Text(result.status.value, style=colour)
            code_text = Text(
                str(result.status_code) if result.status_code is not None else "-",
                style=colour,
            )
            time_text = Text(f"{result.duration_ms:.0f} ms")

            # Build detail column: error message or first failure summary.
            detail = ""
            if result.error_message:
                detail = _truncate(result.error_message)
            elif result.failures:
                first = result.failures[0]
                detail = _truncate(f"{first.assertion}: expected {first.expected}, got {first.actual}")

            detail_text = Text(detail, style="dim" if result.passed else colour)

            table.add_row(icon_text, name_text, status_text, code_text, time_text, detail_text)

            # Show assertion details for failures, indented below the row.
            if result.failures and result.status == TestStatus.FAILED:
                for failure in result.failures:
                    failure_detail = Text()
                    failure_detail.append("    ")
                    failure_detail.append(f"{failure.assertion}: ", style="bold " + colour)
                    failure_detail.append(f"expected {failure.expected}", style="green")
                    failure_detail.append(" | ", style="dim")
                    failure_detail.append(f"actual {failure.actual}", style=colour)
                    table.add_row("", failure_detail, "", "", "", "")

        console.print(table)
        console.print()

    # --- Critical failures callout --------------------------------------------
    if suite.critical_failures:
        critical_lines: list[str] = []
        for cf in suite.critical_failures:
            critical_lines.append(f"  [bold red]{_STATUS_ICONS[TestStatus.FAILED]}[/bold red] {cf.test_name}")
            if cf.failures:
                for f in cf.failures:
                    critical_lines.append(f"    {f.assertion}: expected {f.expected}, got {f.actual}")
            if cf.error_message:
                critical_lines.append(f"    {cf.error_message}")

        console.print(Panel(
            "\n".join(critical_lines),
            title="[bold red]Critical Failures[/bold red]",
            border_style="red",
            padding=(0, 1),
        ))
        console.print()


# ---------------------------------------------------------------------------
# JSON report builder
# ---------------------------------------------------------------------------


def _build_json_report(suite: TestSuite) -> str:
    """Serialise *suite* to a JSON string with an embedded summary.

    The output is a JSON object with two top-level keys:

    * ``summary`` --- aggregate statistics.
    * ``suite`` --- the full Pydantic model dump.

    Returns
    -------
    str
        Pretty-printed JSON.
    """
    summary = _build_summary_dict(suite)
    suite_data = json.loads(suite.model_dump_json())
    output = {
        "summary": summary,
        "suite": suite_data,
    }
    return json.dumps(output, indent=2, default=str)


# ---------------------------------------------------------------------------
# Reporter
# ---------------------------------------------------------------------------


class Reporter:
    """Format and display test suite results.

    Supports two output formats:

    * ``"text"`` --- rich terminal output with tables, colours, and progress
      bars (the default).
    * ``"json"`` --- machine-readable JSON including a summary block.

    Parameters
    ----------
    format:
        Output format, either ``"text"`` or ``"json"``.
    """

    def __init__(self, format: str = "text") -> None:  # noqa: A002
        self._format = format
        self._console = Console()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self, suite: TestSuite) -> str:
        """Generate a formatted report string.

        Parameters
        ----------
        suite:
            The completed test suite to report on.

        Returns
        -------
        str
            The full report as a string.  For ``"text"`` format this is the
            rendered console output (without ANSI escapes); for ``"json"``
            format it is valid JSON.
        """
        if self._format == "json":
            return _build_json_report(suite)

        # Render text format into a string via an in-memory console.
        string_console = Console(record=True, width=120, force_terminal=False)
        _build_text_report(suite, string_console)
        return string_console.export_text()

    def print_report(self, suite: TestSuite) -> None:
        """Print the full report directly to the terminal.

        Uses :mod:`rich` for colourful, structured output when the format
        is ``"text"``.  For ``"json"`` format the JSON string is printed
        with syntax highlighting.

        Parameters
        ----------
        suite:
            The completed test suite to report on.
        """
        if self._format == "json":
            output = _build_json_report(suite)
            self._console.print_json(output)
        else:
            _build_text_report(suite, self._console)

    def print_progress(self, result: TestResult) -> None:
        """Print a single test result as it completes.

        Designed to be passed as the ``on_result`` callback to
        :meth:`TestRunner.run` for live, incremental output.

        Parameters
        ----------
        result:
            The just-completed test result.
        """
        icon = _STATUS_ICONS.get(result.status, "?")
        colour = _STATUS_COLOURS.get(result.status, "white")
        severity_colour = _SEVERITY_COLOURS.get(result.severity, "white")

        line = Text()
        line.append(f" {icon} ", style=colour)
        line.append(f"[{result.severity.value.upper():>8s}] ", style=severity_colour)
        line.append(result.test_name)
        line.append(f"  {result.duration_ms:.0f} ms", style="dim")

        if result.status_code is not None:
            line.append(f"  [{result.status_code}]", style=colour)

        if result.error_message:
            line.append(f"  {_truncate(result.error_message, 50)}", style=colour)
        elif result.failures:
            first = result.failures[0]
            line.append(
                f"  {_truncate(first.assertion + ': ' + first.actual, 50)}",
                style=colour,
            )

        self._console.print(line)
