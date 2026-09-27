"""JSON file-based registry store for agentspec.

Provides persistent storage of validated API specifications and their test
results.  Data lives in a single ``registry.json`` file (default location
``~/.agentspec/registry.json``) and is written atomically to prevent
corruption under concurrent access.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentspec.models.api import APIEndpoint, APISpec
from agentspec.models.registry import (
    CompatibilityIssue,
    CompatibilityResult,
    RegistryEntry,
)

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_DIR = Path.home() / ".agentspec"
_REGISTRY_FILENAME = "registry.json"

# Weights used when scoring per-endpoint compatibility.
_WEIGHT_PARAMS = 0.50
_WEIGHT_REQUEST_BODY = 0.25
_WEIGHT_RESPONSE = 0.25


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _slugify(name: str, version: str) -> str:
    """Generate a stable slug ID from *name* and *version*.

    The result is lowercase, with spaces replaced by dashes, all characters
    outside ``[a-z0-9-]`` stripped, and consecutive dashes collapsed.
    """
    raw = f"{name}-{version}".lower()
    raw = raw.replace(" ", "-")
    raw = re.sub(r"[^a-z0-9\-]", "", raw)
    raw = re.sub(r"-+", "-", raw)
    return raw.strip("-")


def _normalize_path(path: str) -> str:
    """Normalize an endpoint path by replacing path parameters with a fixed placeholder.

    ``/users/{user_id}`` and ``/users/{id}`` both become ``/users/{param}``,
    so they match during compatibility comparison.
    """
    return re.sub(r"\{[^}]+\}", "{param}", path)


def _schema_diff_keys(a: dict[str, Any], b: dict[str, Any]) -> set[str]:
    """Return the top-level keys whose values differ between two dicts."""
    all_keys = set(a) | set(b)
    return {k for k in all_keys if a.get(k) != b.get(k)}


# ---------------------------------------------------------------------------
# RegistryStore
# ---------------------------------------------------------------------------


class RegistryStore:
    """JSON file-based registry for storing and querying API specifications.

    Each registered API is keyed by a slug derived from its name and version
    (e.g. ``my-api-1-0-0``).  The backing file is rewritten atomically on
    every mutation so that readers never see a partially-written state.
    """

    def __init__(self, registry_dir: str | None = None) -> None:
        """Initialize the registry store.

        Args:
            registry_dir: Directory for registry data.  Defaults to
                ``~/.agentspec/``.
        """
        self._dir = Path(registry_dir) if registry_dir else _DEFAULT_REGISTRY_DIR
        self._registry_path = self._dir / _REGISTRY_FILENAME
        self._ensure_registry()

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _ensure_registry(self) -> None:
        """Create the registry directory and file if they do not exist."""
        self._dir.mkdir(parents=True, exist_ok=True)
        if not self._registry_path.exists():
            self._write_data({"entries": {}})
            logger.info("Created new registry at %s", self._registry_path)

    def _read_data(self) -> dict[str, Any]:
        """Read and return the full registry dict from disk.

        Returns a valid ``{"entries": {...}}`` dict even when the file is
        missing, empty, or corrupt --- the caller can always proceed safely.
        """
        try:
            text = self._registry_path.read_text(encoding="utf-8")
            data = json.loads(text)
            if not isinstance(data, dict) or "entries" not in data:
                logger.warning("Registry file has invalid structure; reinitializing")
                return {"entries": {}}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read registry at %s: %s", self._registry_path, exc)
            return {"entries": {}}

    def _write_data(self, data: dict[str, Any]) -> None:
        """Atomically write *data* to the registry file.

        Writes to a temporary file in the same directory, then uses
        :func:`os.replace` (atomic on POSIX when source and destination are
        on the same filesystem) to swap it into place.
        """
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._dir),
            suffix=".tmp",
            prefix=".registry-",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=str)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, str(self._registry_path))
        except BaseException:
            # Best-effort cleanup of the temp file.
            with _suppress_os_error():
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def register(
        self,
        spec: APISpec,
        pass_rate: float = 0.0,
        tags: list[str] | None = None,
    ) -> RegistryEntry:
        """Register an API specification in the registry.

        An ID is auto-generated from the spec's name and version.

        Args:
            spec: The API specification to register.
            pass_rate: Initial test pass rate in [0.0, 1.0].
            tags: Optional tags for categorization and filtering.

        Returns:
            The newly created :class:`RegistryEntry`.

        Raises:
            ValueError: If an entry with the same derived ID already exists.
        """
        api_id = _slugify(spec.name, spec.version)
        data = self._read_data()

        if api_id in data["entries"]:
            raise ValueError(
                f"API '{spec.name}' version '{spec.version}' is already registered "
                f"(id='{api_id}'). Remove it first to re-register."
            )

        now = datetime.now(timezone.utc)
        entry = RegistryEntry(
            id=api_id,
            name=spec.name,
            version=spec.version,
            description=spec.description,
            spec=spec,
            source_path=spec.source_path,
            last_test_pass_rate=pass_rate,
            last_tested_at=now if pass_rate > 0.0 else None,
            registered_at=now,
            tags=tags or [],
        )

        data["entries"][api_id] = json.loads(entry.model_dump_json())
        self._write_data(data)
        logger.info("Registered API '%s' (id=%s)", spec.name, api_id)
        return entry

    def get(self, api_id: str) -> RegistryEntry | None:
        """Look up a registered API by its ID.

        Args:
            api_id: Unique identifier (slug) of the API.

        Returns:
            The :class:`RegistryEntry` if found, otherwise ``None``.
        """
        data = self._read_data()
        raw = data["entries"].get(api_id)
        if raw is None:
            return None
        try:
            return RegistryEntry.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("Corrupt registry entry for '%s': %s", api_id, exc)
            return None

    def list_all(self) -> list[RegistryEntry]:
        """Return every registered API, newest first.

        Corrupt entries are silently skipped (with a warning logged).
        """
        data = self._read_data()
        entries: list[RegistryEntry] = []
        for api_id, raw in data["entries"].items():
            try:
                entries.append(RegistryEntry.model_validate(raw))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping corrupt entry '%s': %s", api_id, exc)
        entries.sort(key=lambda e: e.registered_at, reverse=True)
        return entries

    def remove(self, api_id: str) -> bool:
        """Remove an API from the registry.

        Args:
            api_id: Unique identifier of the API to remove.

        Returns:
            ``True`` if the entry was removed, ``False`` if it was not found.
        """
        data = self._read_data()
        if api_id not in data["entries"]:
            logger.warning("Cannot remove '%s': not found in registry", api_id)
            return False

        del data["entries"][api_id]
        self._write_data(data)
        logger.info("Removed API '%s' from registry", api_id)
        return True

    def update_test_results(
        self,
        api_id: str,
        pass_rate: float,
    ) -> RegistryEntry | None:
        """Record new test results for a registered API.

        Args:
            api_id: Unique identifier of the API.
            pass_rate: New test pass rate in [0.0, 1.0].

        Returns:
            The updated :class:`RegistryEntry`, or ``None`` if the API was
            not found or the stored entry is corrupt.
        """
        data = self._read_data()
        raw = data["entries"].get(api_id)
        if raw is None:
            logger.warning("Cannot update test results for '%s': not found", api_id)
            return None

        try:
            entry = RegistryEntry.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            logger.error("Corrupt registry entry for '%s': %s", api_id, exc)
            return None

        entry.last_test_pass_rate = pass_rate
        entry.last_tested_at = datetime.now(timezone.utc)

        data["entries"][api_id] = json.loads(entry.model_dump_json())
        self._write_data(data)
        logger.info("Updated test results for '%s': pass_rate=%.2f", api_id, pass_rate)
        return entry

    # ------------------------------------------------------------------
    # Compatibility checking
    # ------------------------------------------------------------------

    def check_compatibility(
        self,
        api_a_id: str,
        api_b_id: str,
    ) -> CompatibilityResult:
        """Check compatibility between two registered APIs.

        Compares endpoints by (method, normalized path).  For each pair of
        matching endpoints the parameters, request-body schema, and response
        schemas are compared.  Missing endpoints score ``0.0``; matched
        endpoints are scored as a weighted average of parameter, body, and
        response compatibility.

        Args:
            api_a_id: ID of the first API.
            api_b_id: ID of the second API.

        Returns:
            A :class:`CompatibilityResult` containing the overall score,
            a list of specific issues, and a boolean ``compatible`` flag
            (``True`` when *score* >= 0.8).

        Raises:
            KeyError: If either API ID is not found in the registry.
        """
        entry_a = self.get(api_a_id)
        if entry_a is None:
            raise KeyError(f"API '{api_a_id}' not found in registry")
        entry_b = self.get(api_b_id)
        if entry_b is None:
            raise KeyError(f"API '{api_b_id}' not found in registry")

        endpoints_a = _endpoint_map(entry_a.spec)
        endpoints_b = _endpoint_map(entry_b.spec)
        all_keys = set(endpoints_a) | set(endpoints_b)

        if not all_keys:
            # Both APIs declare zero endpoints -- trivially compatible.
            return CompatibilityResult(
                api_a_id=api_a_id,
                api_a_name=entry_a.name,
                api_b_id=api_b_id,
                api_b_name=entry_b.name,
                compatible=True,
                score=1.0,
                issues=[],
            )

        issues: list[CompatibilityIssue] = []
        endpoint_scores: list[float] = []

        for key in sorted(all_keys):
            method, norm_path = key
            ep_a = endpoints_a.get(key)
            ep_b = endpoints_b.get(key)
            label = f"{method} {norm_path}"

            if ep_a is None:
                issues.append(
                    CompatibilityIssue(
                        issue_type="missing_endpoint",
                        description=(
                            f"Endpoint {label} exists in '{entry_b.name}' "
                            f"but not in '{entry_a.name}'"
                        ),
                        severity="high",
                        endpoint=label,
                    )
                )
                endpoint_scores.append(0.0)
                continue

            if ep_b is None:
                issues.append(
                    CompatibilityIssue(
                        issue_type="missing_endpoint",
                        description=(
                            f"Endpoint {label} exists in '{entry_a.name}' "
                            f"but not in '{entry_b.name}'"
                        ),
                        severity="high",
                        endpoint=label,
                    )
                )
                endpoint_scores.append(0.0)
                continue

            ep_score, ep_issues = _compare_endpoints(
                ep_a, ep_b, label, entry_a.name, entry_b.name
            )
            endpoint_scores.append(ep_score)
            issues.extend(ep_issues)

        score = round(sum(endpoint_scores) / len(endpoint_scores), 4)

        return CompatibilityResult(
            api_a_id=api_a_id,
            api_a_name=entry_a.name,
            api_b_id=api_b_id,
            api_b_name=entry_b.name,
            compatible=score >= 0.8,
            score=score,
            issues=issues,
        )


# ---------------------------------------------------------------------------
# Internal comparison helpers (module-level for testability)
# ---------------------------------------------------------------------------


def _endpoint_map(spec: APISpec) -> dict[tuple[str, str], APIEndpoint]:
    """Build a lookup mapping ``(method, normalized_path)`` to endpoint."""
    result: dict[tuple[str, str], APIEndpoint] = {}
    for ep in spec.endpoints:
        key = (ep.method.value, _normalize_path(ep.path))
        result[key] = ep
    return result


def _compare_endpoints(
    ep_a: APIEndpoint,
    ep_b: APIEndpoint,
    label: str,
    name_a: str,
    name_b: str,
) -> tuple[float, list[CompatibilityIssue]]:
    """Compare two matching endpoints and return ``(score, issues)``.

    The score is a weighted average of three component scores:

    * **Parameter compatibility** -- type, required flag, and location of
      each parameter by name.
    * **Request-body schema compatibility** -- structural comparison of the
      top-level request body schema when present.
    * **Response schema compatibility** -- comparison of response schemas
      for shared status codes.

    Components that are absent from *both* endpoints are treated as fully
    compatible and excluded from the weighted average so they do not
    inflate the score.
    """
    issues: list[CompatibilityIssue] = []

    # Track (weight, score) pairs for components that are present.
    components: list[tuple[float, float]] = []

    # --- 1. Parameter compatibility -----------------------------------
    param_score, param_issues = _compare_parameters(
        ep_a, ep_b, label, name_a, name_b
    )
    issues.extend(param_issues)
    # Always include (even if both have 0 params, the score is 1.0).
    has_params = bool(ep_a.parameters or ep_b.parameters)
    if has_params:
        components.append((_WEIGHT_PARAMS, param_score))

    # --- 2. Request body schema ---------------------------------------
    body_score, body_issues = _compare_request_bodies(ep_a, ep_b, label)
    issues.extend(body_issues)
    has_body = ep_a.request_body_schema is not None or ep_b.request_body_schema is not None
    if has_body:
        components.append((_WEIGHT_REQUEST_BODY, body_score))

    # --- 3. Response schemas ------------------------------------------
    resp_score, resp_issues = _compare_responses(ep_a, ep_b, label)
    issues.extend(resp_issues)
    has_responses = bool(ep_a.responses or ep_b.responses)
    if has_responses:
        components.append((_WEIGHT_RESPONSE, resp_score))

    # --- Aggregate ----------------------------------------------------
    if components:
        total_weight = sum(w for w, _ in components)
        ep_score = sum(w * s for w, s in components) / total_weight
    else:
        # Neither endpoint has any params, body, or responses.
        ep_score = 1.0

    return round(ep_score, 4), issues


def _compare_parameters(
    ep_a: APIEndpoint,
    ep_b: APIEndpoint,
    label: str,
    name_a: str,
    name_b: str,
) -> tuple[float, list[CompatibilityIssue]]:
    """Score parameter compatibility between two endpoints.

    Returns ``(score, issues)`` where *score* is the mean per-parameter
    compatibility (1.0 when all parameters match perfectly).
    """
    issues: list[CompatibilityIssue] = []

    params_a = {p.name: p for p in ep_a.parameters}
    params_b = {p.name: p for p in ep_b.parameters}
    all_names = sorted(set(params_a) | set(params_b))

    if not all_names:
        return 1.0, issues

    per_param: list[float] = []

    for pname in all_names:
        pa = params_a.get(pname)
        pb = params_b.get(pname)

        if pa is None or pb is None:
            present_api = name_a if pa else name_b
            missing_api = name_b if pa else name_a
            p = pa or pb
            assert p is not None  # one of them must exist
            severity = "high" if p.required else "medium"
            issues.append(
                CompatibilityIssue(
                    issue_type="type_mismatch",
                    description=(
                        f"Parameter '{pname}' exists in '{present_api}' "
                        f"but is missing in '{missing_api}'"
                    ),
                    severity=severity,
                    endpoint=label,
                )
            )
            per_param.append(0.0)
            continue

        # Both endpoints declare this parameter -- compare attributes.
        score = 1.0

        if pa.param_type != pb.param_type:
            issues.append(
                CompatibilityIssue(
                    issue_type="type_mismatch",
                    description=(
                        f"Parameter '{pname}' type differs: "
                        f"{pa.param_type.value} vs {pb.param_type.value}"
                    ),
                    severity="high",
                    endpoint=label,
                )
            )
            score -= 0.5

        if pa.required != pb.required:
            issues.append(
                CompatibilityIssue(
                    issue_type="type_mismatch",
                    description=(
                        f"Parameter '{pname}' required status differs: "
                        f"{pa.required} vs {pb.required}"
                    ),
                    severity="low",
                    endpoint=label,
                )
            )
            score -= 0.2

        if pa.location != pb.location:
            issues.append(
                CompatibilityIssue(
                    issue_type="type_mismatch",
                    description=(
                        f"Parameter '{pname}' location differs: "
                        f"{pa.location.value} vs {pb.location.value}"
                    ),
                    severity="medium",
                    endpoint=label,
                )
            )
            score -= 0.3

        per_param.append(max(score, 0.0))

    return sum(per_param) / len(per_param), issues


def _compare_request_bodies(
    ep_a: APIEndpoint,
    ep_b: APIEndpoint,
    label: str,
) -> tuple[float, list[CompatibilityIssue]]:
    """Score request-body schema compatibility.

    Returns ``(score, issues)``.
    """
    issues: list[CompatibilityIssue] = []
    schema_a = ep_a.request_body_schema
    schema_b = ep_b.request_body_schema

    if schema_a is None and schema_b is None:
        return 1.0, issues

    if (schema_a is None) != (schema_b is None):
        issues.append(
            CompatibilityIssue(
                issue_type="schema_diff",
                description=(
                    f"Request body schema present in "
                    f"{'A' if schema_a else 'B'} but absent in "
                    f"{'B' if schema_a else 'A'}"
                ),
                severity="high",
                endpoint=label,
            )
        )
        return 0.0, issues

    # Both schemas are non-None at this point.
    assert schema_a is not None and schema_b is not None
    if schema_a == schema_b:
        return 1.0, issues

    diff_keys = _schema_diff_keys(schema_a, schema_b)
    if not diff_keys:
        return 1.0, issues

    all_keys = set(schema_a) | set(schema_b)
    issues.append(
        CompatibilityIssue(
            issue_type="schema_diff",
            description=(
                f"Request body schemas differ on keys: "
                f"{', '.join(sorted(diff_keys))}"
            ),
            severity="medium",
            endpoint=label,
        )
    )
    score = 1.0 - (len(diff_keys) / max(len(all_keys), 1))
    return max(score, 0.0), issues


def _compare_responses(
    ep_a: APIEndpoint,
    ep_b: APIEndpoint,
    label: str,
) -> tuple[float, list[CompatibilityIssue]]:
    """Score response schema compatibility across shared status codes.

    Returns ``(score, issues)``.
    """
    issues: list[CompatibilityIssue] = []
    resp_a = {r.status_code: r for r in ep_a.responses}
    resp_b = {r.status_code: r for r in ep_b.responses}

    if not resp_a and not resp_b:
        return 1.0, issues

    all_codes = sorted(set(resp_a) | set(resp_b))
    per_code: list[float] = []

    for code in all_codes:
        ra = resp_a.get(code)
        rb = resp_b.get(code)

        if ra is None or rb is None:
            # One API defines this response code, the other does not.
            # Treat as a minor mismatch -- the endpoint still largely agrees.
            per_code.append(0.5)
            continue

        # Both define this status code.
        if not ra.schema_def and not rb.schema_def:
            per_code.append(1.0)
            continue

        if not ra.schema_def or not rb.schema_def:
            issues.append(
                CompatibilityIssue(
                    issue_type="schema_diff",
                    description=(
                        f"Response schema for status {code}: defined in one "
                        f"API but not the other"
                    ),
                    severity="medium",
                    endpoint=label,
                )
            )
            per_code.append(0.3)
            continue

        if ra.schema_def == rb.schema_def:
            per_code.append(1.0)
            continue

        diff_keys = _schema_diff_keys(ra.schema_def, rb.schema_def)
        if not diff_keys:
            per_code.append(1.0)
            continue

        all_keys = set(ra.schema_def) | set(rb.schema_def)
        issues.append(
            CompatibilityIssue(
                issue_type="schema_diff",
                description=(
                    f"Response schema for status {code} differs on keys: "
                    f"{', '.join(sorted(diff_keys))}"
                ),
                severity="medium",
                endpoint=label,
            )
        )
        code_score = 1.0 - (len(diff_keys) / max(len(all_keys), 1))
        per_code.append(max(code_score, 0.0))

    return sum(per_code) / len(per_code) if per_code else 1.0, issues


# ---------------------------------------------------------------------------
# Tiny context manager to suppress OSError silently (cleanup paths only).
# ---------------------------------------------------------------------------

class _suppress_os_error:  # noqa: N801 -- lowercase intentional, contextlib style
    """Context manager that silently suppresses :class:`OSError`."""

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool:
        return exc_type is not None and issubclass(exc_type, OSError)
