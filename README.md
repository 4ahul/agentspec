<div align="center">

# 🧪 agentspec

**Auto-test and validate any API — built for the agent era.**

[![PyPI version](https://img.shields.io/pypi/v/agentspec-cli.svg?style=flat-square&color=blue)](https://pypi.org/project/agentspec-cli/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)
[![MCP Compatible](https://img.shields.io/badge/MCP-compatible-purple?style=flat-square)](https://modelcontextprotocol.io)

agentspec discovers your API endpoints, generates hundreds of tests automatically, and runs them — all from one command. Works with any backend framework. Integrates into Claude Code, Codex, and Cursor via MCP.

[Installation](#installation) · [Quick start](#quick-start) · [Frameworks](#supported-frameworks) · [MCP server](#mcp-server) · [CLI reference](#cli-reference)

</div>

---

## What it does

```
agentspec test app/main.py --base-url http://localhost:8000
```

```
agentspec v0.1.0

Discovering API endpoints...
  Discovered 9 endpoint(s) in Task Management API
Generating tests...
  Generated 128 test case(s)
Running tests against http://localhost:8000 (concurrency=10)...

 ✓ [CRITICAL] SQL injection in body field 'title'         39 ms  [201]
 ✓ [    HIGH] Wrong method PUT on /tasks                  25 ms  [405]
 ✓ [    HIGH] Content-Type matches expected on GET /tasks  31 ms  [200]
 ✓ [  MEDIUM] Enum value 'todo' for param 'status'        32 ms  [200]
 ✓ [  MEDIUM] Request with all optional params            44 ms  [200]
 ...

128 passed | 0 failed | 0 errors  (4314 ms)
██████████████████████████████ 100.0%
```

**One command. Zero config. 128 tests.**

---

## What gets tested

For every endpoint agentspec discovers, it generates tests across five categories:

| Category | What it checks |
|----------|---------------|
| **Happy path** | Valid requests, all param combos, per-enum-value coverage |
| **Edge cases** | Empty strings, unicode, 10K-char strings, missing optional params |
| **Error handling** | Wrong HTTP methods, wrong types, unknown params, extra body fields |
| **Security** | SQL injection (8 variants), XSS, null bytes, path traversal, 1 MB payload bombs |
| **Schema validation** | JSON content-type, valid JSON body, response structure on every endpoint |

---

## Installation

```bash
pip install agentspec-cli
```

Requires Python 3.10+.

---

## Quick start

```bash
# Start your API
uvicorn app.main:app --reload

# Test it — framework is auto-detected
agentspec test app/main.py --base-url http://localhost:8000
```

Works with any framework:

```bash
# Flask
agentspec test api/app.py --base-url http://localhost:5000

# Express / NestJS / Fastify
agentspec test src/app.ts --base-url http://localhost:3000

# OpenAPI spec
agentspec test openapi.yaml --base-url http://localhost:8000

# Scan an entire directory
agentspec test ./src --base-url http://localhost:3000
```

---

## Supported frameworks

| Language | Framework | How it's discovered |
|----------|-----------|-------------------|
| Python | **FastAPI** | AST analysis |
| Python | **Flask** | AST analysis (Blueprints, MethodView, path converters) |
| Python | **Django REST Framework** | OpenAPI spec |
| JS / TS | **Express.js** | Regex analysis |
| JS / TS | **NestJS** | Decorator + controller analysis |
| JS / TS | **Next.js** (App Router) | Export handler + file path analysis |
| JS / TS | **Fastify** | Regex analysis |
| JS / TS | **Hono** | Regex analysis |
| Any | **OpenAPI / Swagger** | JSON or YAML spec |

---

## MCP server

Add agentspec to Claude Code, Codex, or Cursor so your AI agent can test APIs directly:

```json
{
  "mcpServers": {
    "agentspec": {
      "command": "python",
      "args": ["-m", "agentspec.mcp.server"]
    }
  }
}
```

**Available tools:**

| Tool | What it does |
|------|-------------|
| `test_api` | Discover, generate, and run all tests |
| `discover_api` | Discover endpoints without running tests |
| `register_api` | Test and register an API in the local registry |
| `list_registered` | List all registered APIs |
| `check_compatibility` | Detect breaking changes between API versions |

**Example — Claude Code:**
```
You: test my API at http://localhost:8000 using app/main.py

Claude: [calls test_api]

→ 128 passed, 0 failed. Here's the full report...
```

---

## CLI reference

### `agentspec test`

```bash
agentspec test <source> --base-url <url> [options]

Options:
  --base-url TEXT      Base URL of the running API  [required]
  --concurrency INT    Parallel test runners (default: 10)
  --timeout FLOAT      Per-request timeout in seconds (default: 10.0)
  --format TEXT        text | json | html | junit (default: text)
  --output PATH        Write report to file
  --autofix            Show auto-fix suggestions for failures
  --watch              Re-run on source file changes
  --config PATH        Path to .agentspec.yaml
```

### `agentspec discover`
Show all endpoints without running tests.
```bash
agentspec discover app/main.py
```

### `agentspec init`
Generate a `.agentspec.yaml` config file.
```bash
agentspec init
```

### `agentspec register`
Test and register an API in the local registry.
```bash
agentspec register app/main.py --base-url http://localhost:8000 --name "My API" --version 1.0.0
```

### `agentspec compat`
Check for breaking changes between registered versions.
```bash
agentspec compat my-api:1.0.0 my-api:2.0.0
```

### `agentspec serve`
Start the MCP server.
```bash
agentspec serve
```

---

## HTML reports

```bash
agentspec test app/main.py --base-url http://localhost:8000 --format html --output report.html
```

Self-contained single-file HTML with donut chart, summary cards, sortable/filterable test table, and dark/light mode. No external dependencies.

---

## Watch mode

Re-runs the full test suite whenever a source file changes:

```bash
agentspec test app/main.py --base-url http://localhost:8000 --watch
```

---

## Auto-fix suggestions

```bash
agentspec test app/main.py --base-url http://localhost:8000 --autofix
```

When tests fail, the auto-fix engine analyzes failures and generates structured suggestions with code snippets — ready to apply directly or hand to your AI agent.

---

## Config file

```yaml
# .agentspec.yaml
base_url: http://localhost:8000
concurrency: 10
timeout: 10.0
format: text

skip_endpoints:
  - GET /health
  - GET /metrics

skip_categories:
  - security

# Env var expansion supported
headers:
  Authorization: "Bearer ${API_TOKEN}"
  X-API-Key: "${API_KEY}"
```

---

## CI integration

```yaml
# .github/workflows/api-tests.yml
name: API Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install agentspec-cli fastapi uvicorn
      - run: uvicorn app.main:app --host 0.0.0.0 --port 8000 &
      - run: sleep 2
      - run: agentspec test app/main.py --base-url http://localhost:8000 --format junit --output results.xml
      - uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: results.xml
```

---

## API registry

agentspec keeps a local registry at `~/.agentspec/registry.json`. Register multiple versions and diff them:

```bash
# Register v1
agentspec register app/main.py --base-url http://localhost:8000 --name myapi --version 1.0.0

# Register v2 after making changes
agentspec register app/main.py --base-url http://localhost:8001 --name myapi --version 2.0.0

# See what changed
agentspec compat myapi:1.0.0 myapi:2.0.0
```

---

## Claude Code skill

Install the agentspec skill to teach Claude Code to automatically test APIs it generates:

```bash
cp -r skill ~/.claude/skills/agentspec
```

Once installed, Claude will suggest running agentspec after creating new endpoints and use auto-fix output to repair failures itself.

---

## License

MIT
