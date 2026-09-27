.PHONY: install dev sample test lint selftest serve clean

# Install agentspec
install:
	pip install -e .

# Install with dev dependencies
dev:
	pip install -e ".[dev]"

# Start the sample API on port 8000
sample:
	uvicorn examples.sample_api.main:app --port 8000 --reload

# Run agentspec against the sample API (sample must be running)
selftest:
	agentspec test examples/sample_api/main.py --base-url http://localhost:8000

# Discover endpoints in the sample API
discover:
	agentspec discover examples/sample_api/main.py

# Run the unit tests
test:
	pytest tests/ -v --tb=short

# Lint the code
lint:
	ruff check src/ tests/ examples/
	mypy src/agentspec/

# Start the MCP server
serve:
	agentspec serve

# Clean build artifacts
clean:
	rm -rf dist/ build/ *.egg-info src/*.egg-info .mypy_cache .ruff_cache .pytest_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
