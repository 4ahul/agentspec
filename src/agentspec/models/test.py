"""Models for test cases, results, and suites."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from agentspec.models.api import HTTPMethod


class TestCategory(str, Enum):
    HAPPY_PATH = "happy_path"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"
    SECURITY = "security"
    SCHEMA_VALIDATION = "schema_validation"
    PERFORMANCE = "performance"


class TestSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class TestStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class TestCase(BaseModel):
    """A single test case to execute against an API endpoint."""

    id: str
    name: str
    description: str = ""
    category: TestCategory
    severity: TestSeverity = TestSeverity.MEDIUM
    endpoint_path: str
    method: HTTPMethod
    url_params: dict[str, Any] = Field(default_factory=dict)
    query_params: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    body: Any = None
    expected_status: int | list[int] = 200
    expected_body_contains: list[str] = Field(default_factory=list)
    expected_body_not_contains: list[str] = Field(default_factory=list)
    expected_schema: dict[str, Any] | None = None
    timeout_seconds: float = 10.0

    @property
    def expected_statuses(self) -> list[int]:
        if isinstance(self.expected_status, list):
            return self.expected_status
        return [self.expected_status]


class AssertionFailure(BaseModel):
    """Detail about a specific assertion that failed."""

    assertion: str
    expected: str
    actual: str


class TestResult(BaseModel):
    """Result of executing a single test case."""

    test_id: str
    test_name: str
    category: TestCategory
    severity: TestSeverity
    status: TestStatus
    duration_ms: float = 0.0
    status_code: int | None = None
    response_body: Any = None
    failures: list[AssertionFailure] = Field(default_factory=list)
    error_message: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def passed(self) -> bool:
        return self.status == TestStatus.PASSED


class TestSuite(BaseModel):
    """A collection of test cases with summary statistics."""

    api_name: str
    api_source: str = ""
    test_cases: list[TestCase] = Field(default_factory=list)
    results: list[TestResult] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed / len(self.results) * 100

    @property
    def duration_ms(self) -> float:
        return sum(r.duration_ms for r in self.results)

    @property
    def critical_failures(self) -> list[TestResult]:
        return [
            r
            for r in self.results
            if r.status == TestStatus.FAILED and r.severity == TestSeverity.CRITICAL
        ]

    def by_category(self) -> dict[TestCategory, list[TestResult]]:
        groups: dict[TestCategory, list[TestResult]] = {}
        for r in self.results:
            groups.setdefault(r.category, []).append(r)
        return groups
