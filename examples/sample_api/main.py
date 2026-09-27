"""Task Management API -- sample target for agentspec test generation.

Run with:
    uvicorn examples.sample_api.main:app --port 8000
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    """Task priority levels."""
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class Status(str, Enum):
    """Task lifecycle statuses."""
    todo = "todo"
    in_progress = "in_progress"
    done = "done"
    cancelled = "cancelled"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class TaskCreate(BaseModel):
    """Payload for creating a new task."""
    title: str = Field(..., min_length=1, max_length=200, description="Task title")
    description: str | None = Field(None, description="Optional longer description")
    priority: Priority = Field(Priority.medium, description="Task priority")
    due_date: date | None = Field(None, description="Optional due date (YYYY-MM-DD)")


class TaskUpdate(BaseModel):
    """Payload for updating an existing task.  All fields optional."""
    title: str | None = Field(None, min_length=1, max_length=200, description="New title")
    description: str | None = Field(None, description="New description")
    status: Status | None = Field(None, description="New status")
    priority: Priority | None = Field(None, description="New priority")


class TaskResponse(BaseModel):
    """Full task representation returned to clients."""
    id: int
    title: str
    description: str | None = None
    priority: Priority
    status: Status
    due_date: date | None = None
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class TagsRequest(BaseModel):
    """Payload for adding tags to a task."""
    tags: list[str] = Field(..., min_length=1, description="One or more tag strings to add")


class TaskListResponse(BaseModel):
    """Paginated list of tasks."""
    tasks: list[TaskResponse]
    total: int
    limit: int
    offset: int


class StatsResponse(BaseModel):
    """Aggregate statistics about all tasks."""
    total_tasks: int
    by_status: dict[str, int]
    by_priority: dict[str, int]


class MessageResponse(BaseModel):
    """Generic message envelope."""
    detail: str


# ---------------------------------------------------------------------------
# Application & in-memory store
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Task Management API",
    description=(
        "A lightweight task-management service used as a demonstration and test "
        "target for **agentspec**.  Stores tasks in memory -- no database required."
    ),
    version="1.0.0",
    openapi_tags=[
        {"name": "tasks", "description": "CRUD operations on tasks"},
        {"name": "search", "description": "Search and filter tasks"},
        {"name": "tags", "description": "Manage task tags"},
        {"name": "stats", "description": "Aggregate statistics"},
        {"name": "health", "description": "Health / info endpoint"},
    ],
)

_tasks: dict[int, dict] = {}
_next_id: int = 1


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _task_to_response(t: dict) -> TaskResponse:
    return TaskResponse(**t)


# ---------------------------------------------------------------------------
# Seed data
# ---------------------------------------------------------------------------

def _seed() -> None:
    global _next_id
    seeds = [
        {
            "title": "Set up CI pipeline",
            "description": "Configure GitHub Actions for lint, test, and publish.",
            "priority": Priority.high,
            "status": Status.in_progress,
            "due_date": date(2026, 10, 1),
            "tags": ["devops", "ci"],
        },
        {
            "title": "Write unit tests for auth module",
            "description": "Cover login, token refresh, and logout flows.",
            "priority": Priority.critical,
            "status": Status.todo,
            "due_date": date(2026, 10, 5),
            "tags": ["testing"],
        },
        {
            "title": "Update README with examples",
            "description": None,
            "priority": Priority.low,
            "status": Status.todo,
            "due_date": None,
            "tags": [],
        },
        {
            "title": "Fix pagination off-by-one bug",
            "description": "The last page returns one extra item.",
            "priority": Priority.medium,
            "status": Status.done,
            "due_date": date(2026, 9, 20),
            "tags": ["bug"],
        },
    ]
    now = _now()
    for s in seeds:
        task_id = _next_id
        _next_id += 1
        _tasks[task_id] = {
            "id": task_id,
            "title": s["title"],
            "description": s["description"],
            "priority": s["priority"],
            "status": s["status"],
            "due_date": s["due_date"],
            "tags": list(s["tags"]),
            "created_at": now,
            "updated_at": now,
        }


_seed()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"], response_model=dict)
def root() -> dict:
    """Return basic API information and health status.

    Useful as a quick health-check endpoint.  Always returns HTTP 200 when the
    service is running.
    """
    return {
        "name": "Task Management API",
        "version": "1.0.0",
        "status": "healthy",
        "task_count": len(_tasks),
    }


@app.get("/tasks", tags=["tasks"], response_model=TaskListResponse)
def list_tasks(
    status: Status | None = Query(None, description="Filter by status"),
    limit: Annotated[int, Query(ge=1, le=100, description="Max items to return")] = 20,
    offset: Annotated[int, Query(ge=0, description="Number of items to skip")] = 0,
) -> TaskListResponse:
    """List all tasks with optional filtering and pagination.

    Supports filtering by *status* and standard *limit*/*offset* pagination.
    Returns the matching tasks together with the total count (before pagination)
    so clients can build paging controls.
    """
    items = list(_tasks.values())
    if status is not None:
        items = [t for t in items if t["status"] == status]

    total = len(items)
    page = items[offset : offset + limit]
    return TaskListResponse(
        tasks=[_task_to_response(t) for t in page],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.post("/tasks", tags=["tasks"], response_model=TaskResponse, status_code=201)
def create_task(body: TaskCreate) -> TaskResponse:
    """Create a new task.

    Accepts a title (required), optional description, priority, and due date.
    Returns the full task object including the server-assigned *id* and
    timestamps.  Responds with HTTP 201 on success.
    """
    global _next_id
    now = _now()
    task_id = _next_id
    _next_id += 1
    task = {
        "id": task_id,
        "title": body.title,
        "description": body.description,
        "priority": body.priority,
        "status": Status.todo,
        "due_date": body.due_date,
        "tags": [],
        "created_at": now,
        "updated_at": now,
    }
    _tasks[task_id] = task
    return _task_to_response(task)


@app.get("/tasks/search", tags=["search"], response_model=list[TaskResponse])
def search_tasks(
    q: str | None = Query(None, description="Text to search in title and description"),
    status: Status | None = Query(None, description="Filter by status"),
    priority: Priority | None = Query(None, description="Filter by priority"),
) -> list[TaskResponse]:
    """Search tasks by text query, status, and/or priority.

    The *q* parameter performs a case-insensitive substring match against the
    task title and description.  All filters are combined with AND logic.
    """
    results = list(_tasks.values())

    if q is not None:
        q_lower = q.lower()
        results = [
            t for t in results
            if q_lower in t["title"].lower()
            or (t["description"] and q_lower in t["description"].lower())
        ]

    if status is not None:
        results = [t for t in results if t["status"] == status]

    if priority is not None:
        results = [t for t in results if t["priority"] == priority]

    return [_task_to_response(t) for t in results]


@app.get("/tasks/{task_id}", tags=["tasks"], response_model=TaskResponse)
def get_task(task_id: int) -> TaskResponse:
    """Retrieve a single task by its ID.

    Returns HTTP 404 if no task with the given ID exists.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return _task_to_response(_tasks[task_id])


@app.put("/tasks/{task_id}", tags=["tasks"], response_model=TaskResponse)
def update_task(task_id: int, body: TaskUpdate) -> TaskResponse:
    """Update an existing task.

    Only the fields included in the request body are changed; omitted fields
    retain their current values.  Returns HTTP 404 if the task does not exist.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]
    update_data = body.model_dump(exclude_unset=True)

    for field, value in update_data.items():
        task[field] = value

    task["updated_at"] = _now()
    return _task_to_response(task)


@app.delete(
    "/tasks/{task_id}",
    tags=["tasks"],
    response_model=MessageResponse,
)
def delete_task(task_id: int) -> MessageResponse:
    """Delete a task by its ID.

    Returns HTTP 404 if the task does not exist.  On success returns a
    confirmation message.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    del _tasks[task_id]
    return MessageResponse(detail=f"Task {task_id} deleted")


@app.post(
    "/tasks/{task_id}/tags",
    tags=["tags"],
    response_model=TaskResponse,
)
def add_tags(task_id: int, body: TagsRequest) -> TaskResponse:
    """Add one or more tags to an existing task.

    Duplicate tags are silently ignored -- each tag appears at most once.
    Returns HTTP 404 if the task does not exist.
    """
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    task = _tasks[task_id]
    existing = set(task["tags"])
    for tag in body.tags:
        if tag not in existing:
            task["tags"].append(tag)
            existing.add(tag)

    task["updated_at"] = _now()
    return _task_to_response(task)


@app.get("/stats", tags=["stats"], response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Return aggregate statistics about all tasks.

    Includes total count plus breakdowns by status and by priority.  Useful for
    dashboard widgets and monitoring.
    """
    by_status: dict[str, int] = {s.value: 0 for s in Status}
    by_priority: dict[str, int] = {p.value: 0 for p in Priority}

    for t in _tasks.values():
        by_status[t["status"].value] += 1
        by_priority[t["priority"].value] += 1

    return StatsResponse(
        total_tasks=len(_tasks),
        by_status=by_status,
        by_priority=by_priority,
    )
