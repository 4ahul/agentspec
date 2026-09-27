"""Models for API specifications discovered by agentspec."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"
    HEAD = "HEAD"
    OPTIONS = "OPTIONS"


class ParamLocation(str, Enum):
    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    COOKIE = "cookie"
    BODY = "body"


class ParamType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FILE = "file"
    ANY = "any"


class APIParameter(BaseModel):
    """A single parameter of an API endpoint."""

    name: str
    location: ParamLocation
    param_type: ParamType = ParamType.STRING
    required: bool = True
    default: Any = None
    description: str = ""
    constraints: dict[str, Any] = Field(default_factory=dict)
    # constraints can include: min_length, max_length, pattern, minimum, maximum,
    # enum_values, item_type, etc.

    @property
    def example_value(self) -> Any:
        """Generate a plausible example value based on type and constraints."""
        if self.default is not None:
            return self.default
        if "enum_values" in self.constraints and self.constraints["enum_values"]:
            return self.constraints["enum_values"][0]

        match self.param_type:
            case ParamType.STRING:
                return "test_string"
            case ParamType.INTEGER:
                return self.constraints.get("minimum", 1)
            case ParamType.FLOAT:
                return self.constraints.get("minimum", 1.0)
            case ParamType.BOOLEAN:
                return True
            case ParamType.ARRAY:
                return []
            case ParamType.OBJECT:
                return {}
            case _:
                return "test"


class ResponseSchema(BaseModel):
    """Expected response schema for an endpoint."""

    status_code: int = 200
    content_type: str = "application/json"
    schema_def: dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class APIEndpoint(BaseModel):
    """A single API endpoint with full specification."""

    path: str
    method: HTTPMethod
    summary: str = ""
    description: str = ""
    parameters: list[APIParameter] = Field(default_factory=list)
    request_body_schema: dict[str, Any] | None = None
    responses: list[ResponseSchema] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    requires_auth: bool = False
    deprecated: bool = False

    @property
    def id(self) -> str:
        return f"{self.method.value} {self.path}"

    @property
    def path_params(self) -> list[APIParameter]:
        return [p for p in self.parameters if p.location == ParamLocation.PATH]

    @property
    def query_params(self) -> list[APIParameter]:
        return [p for p in self.parameters if p.location == ParamLocation.QUERY]

    @property
    def body_params(self) -> list[APIParameter]:
        return [p for p in self.parameters if p.location == ParamLocation.BODY]


class FrameworkType(str, Enum):
    FASTAPI = "fastapi"
    FLASK = "flask"
    DJANGO = "django"
    EXPRESS = "express"
    NESTJS = "nestjs"
    NEXTJS = "nextjs"
    OPENAPI = "openapi"
    MCP = "mcp"
    UNKNOWN = "unknown"


class APISpec(BaseModel):
    """Complete specification of a discovered API."""

    name: str = "Untitled API"
    version: str = "0.0.0"
    description: str = ""
    base_url: str = ""
    framework: FrameworkType = FrameworkType.UNKNOWN
    endpoints: list[APIEndpoint] = Field(default_factory=list)
    source_path: str = ""
    openapi_version: str = ""
    auth_schemes: list[str] = Field(default_factory=list)

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoints)

    def get_endpoint(self, method: HTTPMethod, path: str) -> APIEndpoint | None:
        for ep in self.endpoints:
            if ep.method == method and ep.path == path:
                return ep
        return None
