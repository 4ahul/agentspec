"""Typer CLI for agentspec -- auto-test and validate agent-generated APIs.

This is the main user-facing interface.  Every command is a thin orchestration
layer that wires together the discovery, generation, runner, reporter, and
registry modules.
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentspec import __version__
from agentspec.models.test import TestCategory

# ---------------------------------------------------------------------------
# App / console setup
# ---------------------------------------------------------------------------

console = Console()

app = typer.Typer(
    name="agentspec",
    help="[bold blue]agentspec[/bold blue] -- Auto-test and validate agent-generated APIs.",
    rich_markup_mode="rich",
    no_args_is_help=True,
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_BANNER = f"[bold blue]agentspec[/bold blue] v{__version__}"


def _print_banner() -> None:
    """Print the agentspec banner line."""
    console.print(_BANNER)
    console.print()


def _parse_categories(raw: str | None) -> list[TestCategory] | None:
    """Parse a comma-separated string of category names into enum values.

    Returns ``None`` (meaning "all categories") when *raw* is ``None`` or
    empty.  Raises :class:`typer.BadParameter` for unrecognised names.
    """
    if not raw:
        return None

    valid_names = {c.value: c for c in TestCategory}
    categories: list[TestCategory] = []

    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in valid_names:
            raise typer.BadParameter(
                f"Unknown category '{token}'.  "
                f"Valid categories: {', '.join(sorted(valid_names))}"
            )
        categories.append(valid_names[token])

    return categories or None


def _parse_tags(raw: str | None) -> list[str]:
    """Split a comma-separated tag string into a list."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Version callback
# ---------------------------------------------------------------------------


def _version_callback(value: bool) -> None:
    """Print version and exit when ``--version`` is passed."""
    if value:
        console.print(f"agentspec {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Show the version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """agentspec -- Auto-test and validate agent-generated APIs."""


# ---------------------------------------------------------------------------
# test
# ---------------------------------------------------------------------------


@app.command()
def test(
    path: str = typer.Argument(
        ...,
        help="Path to API source code, OpenAPI spec file, or directory.",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        "-b",
        help="Base URL of the running API (reads from .agentspec.yaml if not set).",
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: [bold]text[/bold], [bold]json[/bold], [bold]html[/bold], or [bold]junit[/bold].",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save report to file (auto-detects format from extension if --format not set).",
    ),
    categories: Optional[str] = typer.Option(
        None,
        "--categories",
        "-c",
        help="Comma-separated test categories to run (e.g. happy_path,security).",
    ),
    concurrency: int = typer.Option(
        10,
        "--concurrency",
        "-n",
        help="Maximum number of concurrent requests.",
        min=1,
    ),
    timeout: float = typer.Option(
        30.0,
        "--timeout",
        "-t",
        help="Default request timeout in seconds.",
        min=1.0,
    ),
    autofix: bool = typer.Option(
        True,
        "--autofix/--no-autofix",
        help="Show auto-fix suggestions for failures.",
    ),
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Watch for file changes and re-run tests automatically.",
    ),
) -> None:
    """Discover, generate, and run tests against a live API.

    Analyses the API source at PATH, generates a comprehensive test suite,
    executes every test against --base-url, and prints the results.

    Supports auto-fix suggestions, HTML/JUnit reports, and watch mode.
    """
    from agentspec.core.config import load_config, merge_cli_overrides
    from agentspec.core.discovery import discover
    from agentspec.core.generator import TestGenerator
    from agentspec.core.reporter import Reporter
    from agentspec.core.runner import TestRunner

    _print_banner()

    # -- Load config ----------------------------------------------------------
    config = load_config()
    config = merge_cli_overrides(
        config,
        base_url=base_url,
        format=format,
        concurrency=concurrency,
        timeout=timeout,
    )

    effective_base_url = base_url or config.api.base_url
    if not effective_base_url:
        console.print(
            "[bold red]Error:[/bold red] --base-url is required "
            "(or set api.base_url in .agentspec.yaml)"
        )
        raise typer.Exit(code=1)

    # -- Auto-detect format from output extension -----------------------------
    if output and format == "text":
        ext = output.rsplit(".", 1)[-1].lower() if "." in output else ""
        format_map = {"html": "html", "xml": "junit", "json": "json"}
        if ext in format_map:
            format = format_map[ext]

    parsed_categories = _parse_categories(categories)

    # -- Watch mode -----------------------------------------------------------
    if watch:
        from agentspec.core.watcher import Watcher

        watcher = Watcher(
            path=path,
            base_url=effective_base_url,
            debounce_ms=config.watch.debounce_ms,
            clear_screen=config.watch.clear_screen,
            categories=[c.value for c in parsed_categories] if parsed_categories else None,
            format=format,
        )
        console.print("[dim]Starting watch mode... (Ctrl+C to stop)[/dim]\n")
        try:
            asyncio.run(watcher.start())
        except KeyboardInterrupt:
            console.print("\n[yellow]Watch mode stopped.[/yellow]")
        raise typer.Exit(code=0)

    # -- 1. Discover ----------------------------------------------------------
    try:
        console.print("[dim]Discovering API endpoints...[/dim]")
        spec = asyncio.run(discover(path))
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)

    if not spec.endpoints:
        console.print("[bold red]Error:[/bold red] No API endpoints discovered.")
        raise typer.Exit(code=1)

    console.print(
        f"  Discovered [bold]{spec.endpoint_count}[/bold] endpoint(s) "
        f"in [cyan]{spec.name}[/cyan]"
    )

    # -- 2. Generate ----------------------------------------------------------
    console.print("[dim]Generating tests...[/dim]")
    generator = TestGenerator(categories=parsed_categories)
    suite = generator.generate(spec)
    console.print(
        f"  Generated [bold]{len(suite.test_cases)}[/bold] test case(s)"
    )

    # -- 3. Run ---------------------------------------------------------------
    reporter = Reporter(format="text")
    runner = TestRunner(
        base_url=effective_base_url, concurrency=concurrency, timeout=timeout
    )

    console.print(
        f"[dim]Running tests against [bold]{effective_base_url}[/bold] "
        f"(concurrency={concurrency})...[/dim]"
    )
    console.print()

    try:
        suite = asyncio.run(runner.run(suite, on_result=reporter.print_progress))
    except KeyboardInterrupt:
        console.print("\n[yellow]Test run interrupted.[/yellow]")
        raise typer.Exit(code=130)

    # -- 4. Auto-fix suggestions ----------------------------------------------
    fix_report = None
    if autofix and suite.failed > 0:
        from agentspec.core.fixer import AutoFixer

        console.print("\n[dim]Analyzing failures for fix suggestions...[/dim]")
        fixer = AutoFixer(spec=spec, max_suggestions=config.autofix.max_suggestions)
        fix_report = fixer.analyze(suite)
        if fix_report.suggestions:
            console.print(
                f"  Generated [bold]{fix_report.total_suggestions}[/bold] "
                f"fix suggestion(s)\n"
            )

    # -- 5. Report ------------------------------------------------------------
    console.print()

    if format == "html":
        from agentspec.core.html_reporter import HTMLReporter

        html_reporter = HTMLReporter()
        out_path = output or "agentspec-report.html"
        html_reporter.save(suite, out_path, fix_report=fix_report)
        console.print(f"[green]HTML report saved to:[/green] [bold]{out_path}[/bold]")
    elif format == "junit":
        from agentspec.core.junit_reporter import JUnitReporter

        junit_reporter = JUnitReporter()
        out_path = output or "agentspec-results.xml"
        junit_reporter.save(suite, out_path)
        console.print(f"[green]JUnit XML saved to:[/green] [bold]{out_path}[/bold]")
    elif format == "json":
        reporter_json = Reporter(format="json")
        if output:
            import pathlib

            pathlib.Path(output).write_text(reporter_json.report(suite))
            console.print(f"[green]JSON report saved to:[/green] [bold]{output}[/bold]")
        else:
            reporter_json.print_report(suite)
    else:
        reporter.print_report(suite)

    # -- 5b. Print fix suggestions (text mode) --------------------------------
    if fix_report and fix_report.suggestions and format == "text":
        console.print()
        console.print(
            Panel(
                fix_report.summary,
                title="[bold yellow]Fix Suggestions[/bold yellow]",
                border_style="yellow",
            )
        )
        for i, fix in enumerate(fix_report.suggestions[:10], 1):
            sev_colors = {
                "critical": "red",
                "high": "red",
                "medium": "yellow",
                "low": "cyan",
            }
            sev_color = sev_colors.get(fix.severity, "dim")
            console.print(
                f"\n  [{sev_color}]{fix.severity.upper()}[/{sev_color}] "
                f"[bold]{fix.title}[/bold]"
            )
            console.print(f"  [dim]{fix.endpoint}[/dim]")
            console.print(f"  {fix.description}")
            if fix.code_suggestion:
                console.print(f"  [dim]Suggested fix:[/dim]")
                for line in fix.code_suggestion.split("\n")[:8]:
                    console.print(f"    [green]{line}[/green]")
            console.print(f"  [dim]Agent instruction:[/dim] {fix.agent_instruction}")

    # -- 6. Save to file if --output given (text mode) ------------------------
    if output and format == "text":
        import pathlib

        pathlib.Path(output).write_text(reporter.report(suite))
        console.print(f"\n[green]Report saved to:[/green] [bold]{output}[/bold]")

    # -- 7. Exit code ---------------------------------------------------------
    if suite.critical_failures:
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# discover
# ---------------------------------------------------------------------------


@app.command()
def discover(
    path: str = typer.Argument(
        ...,
        help="Path to API source code, OpenAPI spec file, or directory.",
    ),
    format: str = typer.Option(
        "text",
        "--format",
        "-f",
        help="Output format: [bold]text[/bold] or [bold]json[/bold].",
    ),
) -> None:
    """Discover API endpoints from source code or an OpenAPI spec."""
    from agentspec.core.discovery import discover as run_discover

    _print_banner()

    try:
        spec = asyncio.run(run_discover(path))
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)

    if not spec.endpoints:
        console.print("[yellow]No API endpoints found.[/yellow]")
        raise typer.Exit(code=0)

    # -- JSON output ----------------------------------------------------------
    if format == "json":
        console.print_json(spec.model_dump_json())
        return

    # -- Text table -----------------------------------------------------------
    console.print(
        f"[bold]{spec.name}[/bold]  "
        f"v{spec.version}  "
        f"[dim]({spec.framework.value})[/dim]"
    )
    if spec.description:
        console.print(f"[dim]{spec.description}[/dim]")
    console.print()

    table = Table(
        title=f"{spec.endpoint_count} endpoint(s)",
        show_header=True,
        header_style="bold",
        border_style="blue",
        expand=True,
    )
    table.add_column("Method", style="bold cyan", width=8)
    table.add_column("Path", ratio=3)
    table.add_column("Parameters", ratio=2)
    table.add_column("Auth", width=6, justify="center")

    for ep in spec.endpoints:
        param_strs: list[str] = []
        for p in ep.parameters:
            req_marker = "*" if p.required else ""
            param_strs.append(f"{p.name}{req_marker} ({p.param_type.value})")
        params_display = ", ".join(param_strs) if param_strs else "[dim]-[/dim]"

        auth_display = "[red]Yes[/red]" if ep.requires_auth else "[dim]No[/dim]"

        table.add_row(ep.method.value, ep.path, params_display, auth_display)

    console.print(table)


# ---------------------------------------------------------------------------
# register
# ---------------------------------------------------------------------------


@app.command()
def register(
    path: str = typer.Argument(
        ...,
        help="Path to API source code, OpenAPI spec file, or directory.",
    ),
    base_url: str = typer.Option(
        ...,
        "--base-url",
        "-b",
        help="Base URL of the running API to test against.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        help="Override the API name (defaults to the discovered name).",
    ),
    version: Optional[str] = typer.Option(
        None,
        "--version",
        help="Override the API version (defaults to the discovered version).",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags",
        help="Comma-separated tags for the registration.",
    ),
) -> None:
    """Discover, test, and register an API in the local registry."""
    from agentspec.core.discovery import discover as run_discover
    from agentspec.core.generator import TestGenerator
    from agentspec.core.reporter import Reporter
    from agentspec.core.runner import TestRunner
    from agentspec.registry.store import RegistryStore

    _print_banner()

    # -- Discover -------------------------------------------------------------
    try:
        console.print("[dim]Discovering API endpoints...[/dim]")
        spec = asyncio.run(run_discover(path))
    except FileNotFoundError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(code=130)

    if not spec.endpoints:
        console.print("[bold red]Error:[/bold red] No API endpoints discovered.")
        raise typer.Exit(code=1)

    # Apply overrides.
    if name:
        spec.name = name
    if version:
        spec.version = version

    console.print(
        f"  Discovered [bold]{spec.endpoint_count}[/bold] endpoint(s) "
        f"in [cyan]{spec.name}[/cyan] v{spec.version}"
    )

    # -- Generate & run tests -------------------------------------------------
    console.print("[dim]Running validation tests...[/dim]")
    generator = TestGenerator()
    suite = generator.generate(spec)
    reporter = Reporter(format="text")
    runner = TestRunner(base_url=base_url, concurrency=10, timeout=30.0)

    try:
        suite = asyncio.run(runner.run(suite, on_result=reporter.print_progress))
    except KeyboardInterrupt:
        console.print("\n[yellow]Test run interrupted.[/yellow]")
        raise typer.Exit(code=130)

    pass_rate = suite.pass_rate / 100.0  # Normalize to 0.0-1.0 for the store.
    console.print(f"  Pass rate: [bold]{suite.pass_rate:.1f}%[/bold]")
    console.print()

    if pass_rate <= 0:
        console.print(
            "[bold red]Error:[/bold red] All tests failed. "
            "API will not be registered (pass rate must be > 0%)."
        )
        raise typer.Exit(code=1)

    # -- Register -------------------------------------------------------------
    store = RegistryStore()
    parsed_tags = _parse_tags(tags)

    try:
        entry = store.register(spec=spec, pass_rate=pass_rate, tags=parsed_tags)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"[bold green]Registered![/bold green]\n\n"
            f"  ID:        [bold]{entry.id}[/bold]\n"
            f"  Name:      {entry.name}\n"
            f"  Version:   {entry.version}\n"
            f"  Endpoints: {spec.endpoint_count}\n"
            f"  Pass rate: {suite.pass_rate:.1f}%\n"
            f"  Tags:      {', '.join(entry.tags) if entry.tags else '-'}",
            title="[bold blue]agentspec registry[/bold blue]",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command("list")
def list_apis() -> None:
    """List all APIs registered in the local registry."""
    from agentspec.registry.store import RegistryStore

    _print_banner()
    store = RegistryStore()
    entries = store.list_all()

    if not entries:
        console.print("[dim]No APIs registered yet.[/dim]")
        return

    table = Table(
        title=f"{len(entries)} registered API(s)",
        show_header=True,
        header_style="bold",
        border_style="blue",
        expand=True,
    )
    table.add_column("ID", style="bold cyan", ratio=2)
    table.add_column("Name", ratio=2)
    table.add_column("Version", width=10)
    table.add_column("Endpoints", width=10, justify="right")
    table.add_column("Pass Rate", width=11, justify="right")
    table.add_column("Last Tested", ratio=2)

    for entry in entries:
        ep_count = str(entry.spec.endpoint_count)

        rate_str = f"{entry.last_test_pass_rate * 100:.1f}%"
        if entry.last_test_pass_rate >= 0.9:
            rate_style = "green"
        elif entry.last_test_pass_rate >= 0.7:
            rate_style = "yellow"
        else:
            rate_style = "red"

        last_tested = (
            entry.last_tested_at.strftime("%Y-%m-%d %H:%M")
            if entry.last_tested_at
            else "[dim]-[/dim]"
        )

        table.add_row(
            entry.id,
            entry.name,
            entry.version,
            ep_count,
            f"[{rate_style}]{rate_str}[/{rate_style}]",
            last_tested,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


@app.command()
def remove(
    api_id: str = typer.Argument(
        ...,
        help="ID of the registered API to remove.",
    ),
) -> None:
    """Remove a registered API from the local registry."""
    from agentspec.registry.store import RegistryStore

    _print_banner()
    store = RegistryStore()

    if store.remove(api_id):
        console.print(f"[green]Removed API [bold]{api_id}[/bold] from registry.[/green]")
    else:
        console.print(f"[bold red]Error:[/bold red] API '{api_id}' not found in registry.")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# compat
# ---------------------------------------------------------------------------


@app.command()
def compat(
    api_a_id: str = typer.Argument(
        ...,
        help="ID of the first registered API.",
    ),
    api_b_id: str = typer.Argument(
        ...,
        help="ID of the second registered API.",
    ),
) -> None:
    """Check compatibility between two registered APIs."""
    from agentspec.registry.store import RegistryStore

    _print_banner()
    store = RegistryStore()

    try:
        result = store.check_compatibility(api_a_id, api_b_id)
    except KeyError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    # -- Header ---------------------------------------------------------------
    score_pct = result.score * 100
    if result.compatible:
        status_text = "[bold green]COMPATIBLE[/bold green]"
    else:
        status_text = "[bold red]INCOMPATIBLE[/bold red]"

    if score_pct >= 90:
        score_style = "green"
    elif score_pct >= 70:
        score_style = "yellow"
    else:
        score_style = "red"

    console.print(
        Panel(
            f"  {result.api_a_name} [dim]({api_a_id})[/dim]  "
            f"[bold]<->[/bold]  "
            f"{result.api_b_name} [dim]({api_b_id})[/dim]\n\n"
            f"  Score:  [{score_style}][bold]{score_pct:.1f}%[/bold][/{score_style}]\n"
            f"  Status: {status_text}",
            title="[bold blue]Compatibility Check[/bold blue]",
            border_style="blue",
        )
    )

    # -- Issues table ---------------------------------------------------------
    if result.issues:
        console.print()
        issues_table = Table(
            title=f"{len(result.issues)} issue(s)",
            show_header=True,
            header_style="bold",
            border_style="dim",
            expand=True,
        )
        issues_table.add_column("Type", width=18)
        issues_table.add_column("Endpoint", width=20)
        issues_table.add_column("Severity", width=10)
        issues_table.add_column("Description", ratio=4)

        _SEVERITY_STYLE = {
            "high": "red",
            "medium": "yellow",
            "low": "cyan",
        }

        for issue in result.issues:
            sev_style = _SEVERITY_STYLE.get(issue.severity, "dim")
            issues_table.add_row(
                issue.issue_type,
                issue.endpoint or "[dim]-[/dim]",
                f"[{sev_style}]{issue.severity}[/{sev_style}]",
                issue.description,
            )

        console.print(issues_table)
    else:
        console.print("[green]No compatibility issues found.[/green]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port for the MCP server.",
    ),
    host: str = typer.Option(
        "localhost",
        "--host",
        "-H",
        help="Host to bind the MCP server to.",
    ),
) -> None:
    """Start the agentspec MCP server on stdio."""
    _print_banner()
    console.print(
        f"[dim]Starting agentspec MCP server on stdio "
        f"(host={host}, port={port})...[/dim]"
    )

    try:
        from agentspec.mcp.server import run as run_server
    except ImportError:
        console.print(
            "[bold red]Error:[/bold red] MCP server module not found.  "
            "Ensure agentspec.mcp.server is installed."
        )
        raise typer.Exit(code=1)

    try:
        run_server(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\n[yellow]MCP server stopped.[/yellow]")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init() -> None:
    """Generate a starter .agentspec.yaml config file in the current directory."""
    import pathlib

    from agentspec.core.config import generate_default_config

    _print_banner()

    config_path = pathlib.Path(".agentspec.yaml")
    if config_path.exists():
        console.print(
            "[bold yellow]Warning:[/bold yellow] .agentspec.yaml already exists. "
            "Overwrite? [dim](use --force to skip this prompt)[/dim]"
        )
        # For simplicity, don't overwrite
        raise typer.Exit(code=0)

    config_content = generate_default_config()
    config_path.write_text(config_content)
    console.print(
        f"[green]Created[/green] [bold].agentspec.yaml[/bold] in the current directory.\n"
        f"[dim]Edit it to configure your API base URL, test categories, and more.[/dim]"
    )
