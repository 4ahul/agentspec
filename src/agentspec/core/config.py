"""Configuration system for agentspec.

Loads settings from a ``.agentspec.yaml`` file (searched up the directory tree
from the current working directory) and merges them with CLI flag overrides.

The configuration cascade, from lowest to highest priority:

1. Built-in defaults (Pydantic model defaults)
2. ``.agentspec.yaml`` / ``.agentspec.yml`` file
3. CLI flag overrides via :func:`merge_cli_overrides`

String values support ``${ENV_VAR}`` expansion using :func:`os.environ.get`.
"""

from __future__ import annotations

import logging
import os
import re
import textwrap
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment variable expansion
# ---------------------------------------------------------------------------

_ENV_VAR_RE: re.Pattern[str] = re.compile(r"\$\{([^}]+)\}")


def _expand_env_vars(value: str) -> str:
    """Replace ``${VAR_NAME}`` placeholders with environment variable values.

    Unknown variables are replaced with an empty string.
    """

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    return _ENV_VAR_RE.sub(_replace, value)


def _expand_env_in_data(data: Any) -> Any:
    """Recursively expand ``${VAR}`` in all string values within *data*."""
    if isinstance(data, str):
        return _expand_env_vars(data)
    if isinstance(data, dict):
        return {k: _expand_env_in_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_expand_env_in_data(item) for item in data]
    return data


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class APIConfig(BaseModel):
    """API server connection settings."""

    model_config = ConfigDict(extra="ignore")

    base_url: str = ""
    timeout: float = 30.0
    concurrency: int = 10
    headers: dict[str, str] = {}


class TestingConfig(BaseModel):
    """Test generation and execution settings."""

    model_config = ConfigDict(extra="ignore")

    categories: list[str] = [
        "happy_path",
        "edge_case",
        "error_handling",
        "security",
        "schema_validation",
    ]
    severity_threshold: str = "low"
    fail_on_warning: bool = False
    max_test_duration_ms: int = 5000


class DiscoveryConfig(BaseModel):
    """API discovery file scanning settings."""

    model_config = ConfigDict(extra="ignore")

    include: list[str] = ["**/*.py"]
    exclude: list[str] = ["**/test_*.py", "**/__pycache__/**"]
    framework: str = "auto"


class OutputConfig(BaseModel):
    """Report output settings."""

    model_config = ConfigDict(extra="ignore")

    format: str = "text"
    report_dir: str = "./agentspec-reports"
    badge: bool = True
    colors: bool = True


class WatchConfig(BaseModel):
    """Watch mode settings for continuous re-testing."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    debounce_ms: int = 1000
    clear_screen: bool = True


class AutofixConfig(BaseModel):
    """Auto-fix suggestion settings."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    max_suggestions: int = 10
    include_code_snippets: bool = True


class RegistryConfig(BaseModel):
    """Local registry settings for storing discovered API specs."""

    model_config = ConfigDict(extra="ignore")

    dir: str = "~/.agentspec"
    auto_register: bool = False


class WebhookConfig(BaseModel):
    """Webhook integration settings."""

    model_config = ConfigDict(extra="ignore")

    url: str = ""
    on: list[str] = ["failure", "complete"]


class GitHubConfig(BaseModel):
    """GitHub integration settings."""

    model_config = ConfigDict(extra="ignore")

    create_issues: bool = False
    labels: list[str] = ["agentspec", "api-bug"]


class SlackConfig(BaseModel):
    """Slack integration settings."""

    model_config = ConfigDict(extra="ignore")

    channel: str = ""
    on: list[str] = ["failure"]


class IntegrationsConfig(BaseModel):
    """External integration settings (MCP-based output hooks)."""

    model_config = ConfigDict(extra="ignore")

    webhook: WebhookConfig = WebhookConfig()
    github: GitHubConfig = GitHubConfig()
    slack: SlackConfig = SlackConfig()


class AgentSpecConfig(BaseModel):
    """Top-level agentspec configuration.

    Holds every configuration section and serves as the single source of truth
    for a run.  Instantiate directly for defaults, or load from a YAML file
    via :func:`load_config`.
    """

    model_config = ConfigDict(extra="ignore")

    version: int = 1
    api: APIConfig = APIConfig()
    testing: TestingConfig = TestingConfig()
    discovery: DiscoveryConfig = DiscoveryConfig()
    output: OutputConfig = OutputConfig()
    watch: WatchConfig = WatchConfig()
    autofix: AutofixConfig = AutofixConfig()
    registry: RegistryConfig = RegistryConfig()
    integrations: IntegrationsConfig = IntegrationsConfig()


# ---------------------------------------------------------------------------
# Path expansion helper
# ---------------------------------------------------------------------------


def _expand_paths(config: AgentSpecConfig) -> AgentSpecConfig:
    """Expand ``~`` in path-like fields so downstream code gets absolute paths."""
    config.registry.dir = str(Path(config.registry.dir).expanduser())
    config.output.report_dir = str(Path(config.output.report_dir).expanduser())
    return config


# ---------------------------------------------------------------------------
# Config file resolution
# ---------------------------------------------------------------------------

_CONFIG_FILENAMES: tuple[str, ...] = (".agentspec.yaml", ".agentspec.yml")


def find_config_file(start_dir: str | None = None) -> str | None:
    """Search up the directory tree for ``.agentspec.yaml`` or ``.agentspec.yml``.

    Starting from *start_dir* (defaults to the current working directory), each
    parent directory is checked in order.  The first matching file path is
    returned, or ``None`` if no config file is found (the filesystem root is the
    stop condition).

    Parameters
    ----------
    start_dir:
        Directory to start the upward search from.  Uses ``os.getcwd()`` when
        ``None``.

    Returns
    -------
    str | None
        Absolute path to the config file, or ``None``.
    """
    current = Path(start_dir) if start_dir else Path.cwd()
    current = current.resolve()

    while True:
        for filename in _CONFIG_FILENAMES:
            candidate = current / filename
            if candidate.is_file():
                return str(candidate)

        parent = current.parent
        if parent == current:
            # Reached filesystem root.
            break
        current = parent

    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_config(path: str | None = None) -> AgentSpecConfig:
    """Load configuration from a ``.agentspec.yaml`` file.

    The lookup order is:

    1. If *path* is given explicitly, load that file.
    2. Otherwise, search up from the current working directory using
       :func:`find_config_file`.
    3. If no file is found, return the built-in default configuration.

    Environment variables of the form ``${VAR_NAME}`` are expanded in all
    string values after the YAML is parsed.

    Parameters
    ----------
    path:
        Explicit path to a YAML config file.  When ``None``, the file is
        located automatically.

    Returns
    -------
    AgentSpecConfig
        The fully resolved configuration object.
    """
    config_path: str | None = path

    if config_path is None:
        config_path = find_config_file()

    if config_path is None:
        logger.debug("No .agentspec.yaml found; using default configuration")
        return _expand_paths(AgentSpecConfig())

    logger.info("Loading config from %s", config_path)

    try:
        raw_text = Path(config_path).read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read config file %s: %s", config_path, exc)
        return _expand_paths(AgentSpecConfig())

    try:
        raw_data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        logger.warning(
            "Malformed YAML in %s: %s — falling back to default configuration",
            config_path,
            exc,
        )
        return _expand_paths(AgentSpecConfig())

    if not isinstance(raw_data, dict):
        logger.warning(
            "Expected a mapping at the top level of %s — falling back to default configuration",
            config_path,
        )
        return _expand_paths(AgentSpecConfig())

    # Expand environment variables throughout the parsed data.
    expanded = _expand_env_in_data(raw_data)

    try:
        config = AgentSpecConfig.model_validate(expanded)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Invalid configuration in %s: %s — falling back to default configuration",
            config_path,
            exc,
        )
        return _expand_paths(AgentSpecConfig())

    return _expand_paths(config)


# ---------------------------------------------------------------------------
# CLI override merging
# ---------------------------------------------------------------------------

# Maps flat CLI argument names to dotted paths within AgentSpecConfig.
_CLI_OVERRIDE_MAP: dict[str, str] = {
    "base_url": "api.base_url",
    "timeout": "api.timeout",
    "concurrency": "api.concurrency",
    "format": "output.format",
    "report_dir": "output.report_dir",
    "colors": "output.colors",
    "badge": "output.badge",
    "severity_threshold": "testing.severity_threshold",
    "fail_on_warning": "testing.fail_on_warning",
    "max_test_duration_ms": "testing.max_test_duration_ms",
    "framework": "discovery.framework",
    "watch": "watch.enabled",
    "debounce_ms": "watch.debounce_ms",
    "clear_screen": "watch.clear_screen",
    "autofix": "autofix.enabled",
    "max_suggestions": "autofix.max_suggestions",
    "include_code_snippets": "autofix.include_code_snippets",
    "registry_dir": "registry.dir",
    "auto_register": "registry.auto_register",
}


def merge_cli_overrides(config: AgentSpecConfig, **overrides: Any) -> AgentSpecConfig:
    """Merge CLI flag overrides into a loaded configuration.

    Keyword arguments whose values are ``None`` are ignored.  Known argument
    names are mapped to their corresponding nested config fields using an
    internal lookup table.  Arguments that match a top-level or nested field
    name directly (dot-separated, e.g. ``api.base_url``) are also supported.

    A *new* :class:`AgentSpecConfig` is returned; the original is not mutated.

    Parameters
    ----------
    config:
        The base configuration to overlay.
    **overrides:
        Flat keyword arguments mirroring CLI flags.  Only non-``None`` values
        are applied.

    Returns
    -------
    AgentSpecConfig
        A new configuration with overrides applied.

    Examples
    --------
    >>> cfg = load_config()
    >>> cfg = merge_cli_overrides(cfg, base_url="http://localhost:3000", format="json")
    """
    # Deep-copy via round-trip serialisation so the original stays untouched.
    data = config.model_dump()

    for key, value in overrides.items():
        if value is None:
            continue

        # Resolve the dotted path for this key.
        dotted = _CLI_OVERRIDE_MAP.get(key, key)
        parts = dotted.split(".")

        # Walk into the nested dict and set the value.
        target = data
        for part in parts[:-1]:
            if isinstance(target, dict) and part in target:
                target = target[part]
            else:
                logger.debug("Unknown config override path: %s (from key %r)", dotted, key)
                target = None
                break

        if target is not None and isinstance(target, dict):
            final_key = parts[-1]
            if final_key in target:
                target[final_key] = value
            else:
                logger.debug("Unknown config field %r in override %r", final_key, key)

    result = AgentSpecConfig.model_validate(data)
    return _expand_paths(result)


# ---------------------------------------------------------------------------
# Default config generation
# ---------------------------------------------------------------------------


def generate_default_config() -> str:
    """Generate a well-commented default ``.agentspec.yaml`` file as a string.

    The returned YAML is ready to be written to disk and includes inline
    documentation for every section.

    Returns
    -------
    str
        A complete YAML configuration template.
    """
    return textwrap.dedent("""\
        # agentspec configuration
        version: 1

        # API server settings
        api:
          base_url: "http://localhost:8000"
          timeout: 30
          concurrency: 10
          headers:
            Authorization: "Bearer ${API_TOKEN}"
            X-Custom-Header: "value"

        # Test settings
        testing:
          categories:
            - happy_path
            - edge_case
            - error_handling
            - security
            - schema_validation
          severity_threshold: medium  # skip tests below this severity
          fail_on_warning: false
          max_test_duration_ms: 5000

        # Discovery settings
        discovery:
          include:
            - "src/**/*.py"
            - "app/**/*.py"
          exclude:
            - "**/test_*.py"
            - "**/__pycache__/**"
          framework: auto  # auto, fastapi, flask, openapi, mcp

        # Output settings
        output:
          format: text  # text, json, html, junit
          report_dir: "./agentspec-reports"
          badge: true
          colors: true

        # Watch mode
        watch:
          enabled: false
          debounce_ms: 1000
          clear_screen: true

        # Auto-fix settings
        autofix:
          enabled: true
          max_suggestions: 10
          include_code_snippets: true

        # Registry
        registry:
          dir: "~/.agentspec"
          auto_register: false

        # Integrations (MCP-based output hooks)
        integrations:
          webhook:
            url: ""
            on: ["failure", "complete"]
          github:
            create_issues: false
            labels: ["agentspec", "api-bug"]
          slack:
            channel: ""
            on: ["failure"]
    """)
