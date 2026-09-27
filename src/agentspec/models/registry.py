"""Models for the API registry."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from agentspec.models.api import APISpec


class RegistryEntry(BaseModel):
    """A registered API with its spec and test history."""

    id: str
    name: str
    version: str
    description: str = ""
    spec: APISpec
    source_path: str = ""
    last_test_pass_rate: float = 0.0
    last_tested_at: datetime | None = None
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompatibilityIssue(BaseModel):
    """A specific compatibility issue between two APIs."""

    issue_type: str  # "missing_endpoint", "type_mismatch", "schema_diff"
    description: str
    severity: str = "medium"
    endpoint: str = ""


class CompatibilityResult(BaseModel):
    """Result of checking compatibility between two registered APIs."""

    api_a_id: str
    api_a_name: str
    api_b_id: str
    api_b_name: str
    compatible: bool = False
    score: float = 0.0  # 0.0 to 1.0
    issues: list[CompatibilityIssue] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
