#!/usr/bin/env python3
"""
Self-test script for agentspec.
Starts the sample API, runs agentspec against it, and validates the results.

Usage:
    python scripts/selftest.py
"""

import asyncio
import subprocess
import sys
import time


async def main() -> int:
    print("\n" + "=" * 60)
    print("  agentspec self-test")
    print("=" * 60)

    # Step 1: Start the sample API server
    print("\n[1/5] Starting sample API server on port 8787...")
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "examples.sample_api.main:app",
            "--port", "8787",
            "--log-level", "warning",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    import httpx

    for i in range(30):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get("http://localhost:8787/")
                if resp.status_code == 200:
                    print("  Server is up!")
                    break
        except Exception:
            pass
        await asyncio.sleep(0.5)
    else:
        print("  ERROR: Server failed to start")
        server_proc.kill()
        return 1

    try:
        # Step 2: Test discovery
        print("\n[2/5] Testing API discovery...")
        from agentspec.core.discovery import discover

        spec = await discover("examples/sample_api/main.py")
        print(f"  Found {spec.endpoint_count} endpoints")
        print(f"  Framework: {spec.framework.value}")
        for ep in spec.endpoints:
            params = ", ".join(p.name for p in ep.parameters)
            print(f"    {ep.method.value:6} {ep.path:30} params=[{params}]")

        if spec.endpoint_count == 0:
            print("  ERROR: No endpoints discovered!")
            return 1

        # Step 3: Test generation
        print("\n[3/5] Generating test suite...")
        from agentspec.core.generator import TestGenerator

        generator = TestGenerator()
        suite = generator.generate(spec)
        print(f"  Generated {len(suite.test_cases)} test cases")
        from agentspec.models.test import TestCategory

        for cat in TestCategory:
            count = sum(1 for tc in suite.test_cases if tc.category == cat)
            if count > 0:
                print(f"    {cat.value:20} → {count} tests")

        # Step 4: Run tests
        print("\n[4/5] Running tests against live API...")
        from agentspec.core.runner import TestRunner

        runner = TestRunner(base_url="http://localhost:8787", concurrency=5, timeout=10.0)

        completed = 0
        total = len(suite.test_cases)

        def on_progress(result):  # noqa: ANN001
            nonlocal completed
            completed += 1
            icon = "✓" if result.passed else "✗"
            print(f"  [{completed}/{total}] {icon} {result.test_name[:60]}")

        suite = await runner.run(suite, on_result=on_progress)

        # Step 5: Report
        print("\n[5/5] Results:")
        print(f"  Total:    {suite.total}")
        print(f"  Passed:   {suite.passed}")
        print(f"  Failed:   {suite.failed}")
        print(f"  Errors:   {suite.errors}")
        print(f"  Skipped:  {suite.skipped}")
        print(f"  Pass Rate: {suite.pass_rate:.1f}%")
        print(f"  Duration:  {suite.duration_ms:.0f}ms")

        if suite.failed > 0:
            print("\n  Failures:")
            for r in suite.results:
                if not r.passed and r.failures:
                    print(f"    [{r.category.value}] {r.test_name}")
                    for f in r.failures:
                        print(f"      {f.assertion}: expected={f.expected}, got={f.actual}")

        # Print the rich report too
        print("\n" + "=" * 60)
        print("  Full Report:")
        print("=" * 60)
        from agentspec.core.reporter import Reporter

        reporter = Reporter(format="text")
        reporter.print_report(suite)

        # Step 6: Test registry
        print("\n[Bonus] Testing registry...")
        from agentspec.registry.store import RegistryStore
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            store = RegistryStore(registry_dir=tmpdir)
            entry = store.register(spec, pass_rate=suite.pass_rate, tags=["sample", "tasks"])
            print(f"  Registered: {entry.id} ({entry.name} v{entry.version})")

            all_entries = store.list_all()
            print(f"  Registry contains {len(all_entries)} entries")

            retrieved = store.get(entry.id)
            assert retrieved is not None, "Failed to retrieve registered API"
            print(f"  Retrieved: {retrieved.name}")

        print("\n" + "=" * 60)
        print("  SELF-TEST COMPLETE")
        print("=" * 60 + "\n")

        return 0 if suite.pass_rate > 0 else 1

    finally:
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print("  Server stopped.\n")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
