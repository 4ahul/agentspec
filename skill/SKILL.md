# agentspec — Auto-Test Agent-Generated APIs

## When to Use This Skill

Use this skill **any time you generate, modify, or review API code** — REST endpoints, MCP servers, FastAPI/Flask apps, OpenAPI specs. agentspec automatically discovers endpoints, generates comprehensive tests, runs them, and gives you structured fix suggestions you can apply directly.

**Triggers:**
- You just wrote or modified a FastAPI/Flask/Express API
- You created or updated an MCP server with tool definitions
- You're reviewing API code and want to validate it works
- The user asks you to "test", "validate", or "check" an API
- You generated an OpenAPI spec and want to verify it
- The user asks for API quality assurance

## Prerequisites

agentspec must be installed and configured as an MCP server. If not available, tell the user:

```bash
pip install agentspec
```

**MCP server config** (add to Claude Code / Claude Desktop settings):
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

## How to Use

### After generating API code:

1. **Discover** what you built:
   Call the `discover_api` tool with the path to your API source file.
   This shows you all endpoints, parameters, and schemas found.

2. **Test** it automatically:
   Call the `test_api` tool with the source path and the base URL where the API is running.
   agentspec generates and runs happy path, edge case, error handling, security, and schema validation tests.

3. **Fix** failures:
   Read the test results. For each failure, agentspec provides:
   - What failed and why
   - A fix suggestion with code
   - A direct instruction you can follow to fix it
   
   Apply the fixes, then re-run `test_api` to verify.

4. **Register** (optional):
   Call `register_api` to save validated APIs in the local registry for compatibility checking later.

### The build→test→fix loop:

```
You write API code
    ↓
agentspec discover (find endpoints)
    ↓
agentspec test (generate + run tests)
    ↓
If failures → read fix suggestions → apply fixes → re-test
    ↓
If all pass → done (optionally register)
```

### Testing MCP servers:

agentspec can also test MCP tool definitions. If you've just built an MCP server:
- Point `discover_api` at the server's Python source file
- It will find `@server.list_tools()` and `@server.call_tool()` definitions
- `test_api` generates tests for each tool's input schema

## Available Tools

| Tool | What it does |
|------|-------------|
| `test_api` | Discover + generate + run full test suite against a live API |
| `discover_api` | Discover API endpoints from source code or spec |
| `register_api` | Test and register an API in the local registry |
| `list_registered` | List all APIs in the registry |
| `check_compatibility` | Check if two APIs are compatible |

## CLI Commands (for manual use)

```bash
# Test an API
agentspec test app/main.py --base-url http://localhost:8000

# Discover endpoints only
agentspec discover app/main.py

# Watch mode — re-test on every code change
agentspec test app/main.py --base-url http://localhost:8000 --watch

# Generate HTML report
agentspec test app/main.py --base-url http://localhost:8000 --format html --output report.html

# Generate JUnit XML for CI
agentspec test app/main.py --base-url http://localhost:8000 --format junit --output results.xml

# Start MCP server
agentspec serve
```

## Configuration

Projects can include a `.agentspec.yaml` for default settings:

```yaml
api:
  base_url: "http://localhost:8000"
  timeout: 30
testing:
  categories: [happy_path, edge_case, error_handling, security, schema_validation]
autofix:
  enabled: true
output:
  format: text
```

Generate a starter config: `agentspec init`

## Best Practices

1. **Always test after generating API code.** Don't wait for the user to ask — proactively validate.
2. **Run the full test suite**, not just happy paths. Security and edge case tests catch real bugs.
3. **Apply fix suggestions iteratively.** Fix critical issues first, then re-test.
4. **Register APIs** that pass so you can check compatibility later.
5. **Use watch mode** during development for continuous feedback.
