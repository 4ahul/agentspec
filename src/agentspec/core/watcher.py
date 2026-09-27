"""File watcher that re-runs discovery and tests when API source code changes.

Uses :func:`os.stat` polling instead of OS-specific file-system event APIs so
that it works everywhere without additional dependencies.  When a change is
detected the full pipeline --- discover, generate, run, report --- is
re-executed and the results are printed to the terminal.

Usage::

    watcher = Watcher(
        path="./app/main.py",
        base_url="http://localhost:8000",
        debounce_ms=1000,
    )
    await watcher.start()   # blocks until KeyboardInterrupt
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentspec.models.test import TestCategory

logger = logging.getLogger(__name__)

# File extensions the watcher monitors.
_WATCH_EXTENSIONS: frozenset[str] = frozenset({".py", ".json", ".yaml", ".yml"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_file_mtimes(root: Path) -> dict[str, float]:
    """Walk *root* and return a mapping of file path to modification time.

    Only files whose extension is in :data:`_WATCH_EXTENSIONS` are included.
    Symlinks, hidden directories, and common virtual-environment paths are
    skipped.
    """
    mtimes: dict[str, float] = {}

    if root.is_file():
        # Single file mode.
        if root.suffix in _WATCH_EXTENSIONS:
            try:
                mtimes[str(root)] = os.stat(str(root)).st_mtime
            except OSError:
                pass
        return mtimes

    skip_dirs: frozenset[str] = frozenset({
        "__pycache__", ".git", ".venv", "venv", "node_modules",
        ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "dist", "build", "*.egg-info",
    })

    for dirpath, dirnames, filenames in os.walk(str(root)):
        # Prune hidden and virtual-env directories.
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d not in skip_dirs
        ]

        for fname in filenames:
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _WATCH_EXTENSIONS:
                continue
            full = os.path.join(dirpath, fname)
            try:
                mtimes[full] = os.stat(full).st_mtime
            except OSError:
                pass

    return mtimes


def _detect_changes(
    old: dict[str, float],
    new: dict[str, float],
) -> list[str]:
    """Compare two mtime snapshots and return paths that changed.

    A file counts as changed if:
    * its mtime increased (content modified),
    * it is newly present (created), or
    * it was removed.
    """
    changed: list[str] = []

    for path, mtime in new.items():
        prev = old.get(path)
        if prev is None or mtime > prev:
            changed.append(path)

    # Deleted files.
    for path in old:
        if path not in new:
            changed.append(path)

    return changed


def _parse_categories(raw: list[str] | None) -> list[TestCategory] | None:
    """Convert a list of category name strings to enum values.

    Returns ``None`` (meaning all categories) when *raw* is ``None`` or
    empty.
    """
    if not raw:
        return None

    valid = {c.value: c for c in TestCategory}
    result: list[TestCategory] = []
    for name in raw:
        cat = valid.get(name.strip().lower())
        if cat is not None:
            result.append(cat)
        else:
            logger.warning("Ignoring unknown test category: %r", name)
    return result or None


# ---------------------------------------------------------------------------
# Watcher
# ---------------------------------------------------------------------------


class Watcher:
    """Poll the filesystem and re-run the agentspec pipeline on changes.

    The watcher tracks modification times of ``.py``, ``.json``, and
    ``.yaml`` files under the given *path*.  When a change is detected (after
    a debounce interval) it runs:

    1. :func:`~agentspec.core.discovery.discover` -- find endpoints
    2. :class:`~agentspec.core.generator.TestGenerator` -- generate tests
    3. :class:`~agentspec.core.runner.TestRunner` -- execute tests
    4. :class:`~agentspec.core.reporter.Reporter` -- print results

    Parameters
    ----------
    path:
        Path to watch -- a single file or a directory.
    base_url:
        Root URL of the running API (e.g. ``http://localhost:8000``).
    debounce_ms:
        Minimum quiet time (in milliseconds) after the last detected change
        before a re-run is triggered.  Also used as the polling interval.
    clear_screen:
        Whether to clear the terminal before each re-run.
    categories:
        Optional list of category names to limit test generation.
    format:
        Reporter output format -- ``"text"`` or ``"json"``.
    """

    def __init__(
        self,
        path: str,
        base_url: str,
        debounce_ms: int = 1000,
        clear_screen: bool = True,
        categories: list[str] | None = None,
        format: str = "text",  # noqa: A002
    ) -> None:
        self._path = Path(path).resolve()
        self._base_url = base_url
        self._debounce_s = debounce_ms / 1000.0
        self._clear_screen = clear_screen
        self._categories = _parse_categories(categories)
        self._format = format
        self._mtimes: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start watching for changes.  Blocks until interrupted.

        The first cycle runs immediately so that the user sees results
        without waiting for a file change.  Subsequent cycles are triggered
        only when a file modification is detected.

        Raises
        ------
        KeyboardInterrupt
            Caught internally for a clean shutdown message.
        """
        logger.info(
            "Watching %s (debounce=%dms, format=%s)",
            self._path,
            int(self._debounce_s * 1000),
            self._format,
        )
        print(
            f"\n  agentspec watch  |  {self._path}"
            f"  |  debounce {int(self._debounce_s * 1000)}ms"
            f"  |  Ctrl+C to stop\n"
        )

        # Take an initial snapshot and run the first cycle immediately.
        self._mtimes = _collect_file_mtimes(self._path)
        await self._run_cycle()

        try:
            while True:
                await asyncio.sleep(self._debounce_s)

                new_mtimes = _collect_file_mtimes(self._path)
                changed = _detect_changes(self._mtimes, new_mtimes)

                if not changed:
                    continue

                self._mtimes = new_mtimes

                # Debounce: wait once more, then re-snapshot to catch rapid
                # successive saves.
                await asyncio.sleep(self._debounce_s)
                self._mtimes = _collect_file_mtimes(self._path)

                # Print change summary.
                now = datetime.now(timezone.utc).strftime("%H:%M:%S")
                changed_names = [os.path.basename(p) for p in changed[:5]]
                suffix = f" (+{len(changed) - 5} more)" if len(changed) > 5 else ""
                print(
                    f"\n  [{now}] Change detected: "
                    f"{', '.join(changed_names)}{suffix}\n"
                )

                await self._run_cycle()

        except KeyboardInterrupt:
            print("\n  Watch stopped.\n")

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _run_cycle(self) -> None:
        """Run one discover -> generate -> test -> report cycle.

        Errors are caught and printed so that the watcher keeps running.
        """
        if self._clear_screen:
            print("\033[2J\033[H", end="", flush=True)

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"  agentspec  |  {now}  |  {self._path}\n")

        try:
            # Lazy imports to avoid circular dependencies and to keep the
            # module importable even when httpx/anyio are not installed.
            from agentspec.core.discovery import discover
            from agentspec.core.generator import TestGenerator
            from agentspec.core.reporter import Reporter
            from agentspec.core.runner import TestRunner

            # 1. Discover
            spec = await discover(str(self._path))
            if not spec.endpoints:
                print("  No API endpoints discovered.\n")
                return

            print(
                f"  Discovered {spec.endpoint_count} endpoint(s) "
                f"in {spec.name}\n"
            )

            # 2. Generate
            generator = TestGenerator(categories=self._categories)
            suite = generator.generate(spec)
            print(f"  Generated {len(suite.test_cases)} test case(s)\n")

            # 3. Run
            reporter = Reporter(format=self._format)
            runner = TestRunner(base_url=self._base_url)
            suite = await runner.run(suite, on_result=reporter.print_progress)

            # 4. Report
            print()
            reporter.print_report(suite)

        except KeyboardInterrupt:
            raise  # Let the outer loop handle Ctrl+C.
        except Exception as exc:  # noqa: BLE001
            logger.error("Error during watch cycle: %s", exc, exc_info=True)
            print(f"\n  Error: {exc}\n")
            print("  Waiting for next change...\n")
