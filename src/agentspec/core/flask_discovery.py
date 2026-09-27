"""Flask route discovery via AST analysis.

Parses Python source files that use Flask or Flask Blueprints and extracts
every route definition, its HTTP methods, path parameters, and query/body hints.

Supported patterns
------------------
- ``@app.route('/path', methods=['GET', 'POST'])``
- ``@app.get('/path')``, ``@app.post``, ``@app.put``, ``@app.patch``, ``@app.delete``
- ``Blueprint`` with ``url_prefix`` (prefix is prepended to each route path)
- ``MethodView`` class-based views (get/post/put/patch/delete methods)
- Flask path converters: ``<int:id>``, ``<string:name>``, ``<float:x>``, ``<path:tail>``
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

from agentspec.models.api import (
    APIEndpoint,
    APIParameter,
    APISpec,
    FrameworkType,
    HTTPMethod,
    ParamLocation,
    ParamType,
)

logger = logging.getLogger(__name__)

# Maps Flask converter names to ParamType
_FLASK_CONVERTER_MAP: dict[str, ParamType] = {
    "int": ParamType.INTEGER,
    "float": ParamType.FLOAT,
    "string": ParamType.STRING,
    "str": ParamType.STRING,
    "path": ParamType.STRING,
    "uuid": ParamType.STRING,
    "any": ParamType.ANY,
}

# Flask-style <converter:name> or <name>
_FLASK_PARAM_RE = re.compile(r"<(?:(\w+):)?(\w+)>")


def _flask_path_to_openapi(flask_path: str) -> tuple[str, list[APIParameter]]:
    """Convert a Flask path string to OpenAPI ``{param}`` style.

    Returns the converted path and the list of path parameters.

    Examples
    --------
    ``/users/<int:user_id>`` → ``/users/{user_id}``, [APIParameter(name='user_id', ...)]
    ``/files/<path:filename>`` → ``/files/{filename}``, [...]
    """
    path_params: list[APIParameter] = []
    openapi_path = flask_path

    for match in _FLASK_PARAM_RE.finditer(flask_path):
        converter = match.group(1) or "string"
        param_name = match.group(2)
        param_type = _FLASK_CONVERTER_MAP.get(converter, ParamType.STRING)
        path_params.append(
            APIParameter(
                name=param_name,
                location=ParamLocation.PATH,
                param_type=param_type,
                required=True,
            )
        )
        openapi_path = openapi_path.replace(match.group(0), f"{{{param_name}}}")

    return openapi_path, path_params


def _extract_docstring(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Return the first string constant in a function body as its docstring."""
    if (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    ):
        return node.body[0].value.value.strip()
    return ""


def _methods_from_node(decorator: ast.expr) -> list[HTTPMethod]:
    """Extract HTTP methods from a route decorator.

    Handles:
    - ``@app.route('/path', methods=['GET', 'POST'])``
    - ``@app.get``, ``@app.post``, etc.
    - ``@bp.route(...)``
    """
    if not isinstance(decorator, ast.Call):
        # Plain attribute access like @app.get — method encoded in name
        if isinstance(decorator, ast.Attribute):
            m = decorator.attr.upper()
            if m in {e.value for e in HTTPMethod}:
                return [HTTPMethod(m)]
        return [HTTPMethod.GET]

    func = decorator.func
    # Shorthand: @app.get(...), @app.post(...), etc.
    if isinstance(func, ast.Attribute):
        method_name = func.attr.upper()
        if method_name in {e.value for e in HTTPMethod}:
            return [HTTPMethod(method_name)]

    # @app.route(..., methods=[...])
    for kw in decorator.keywords:
        if kw.arg == "methods" and isinstance(kw.value, ast.List):
            methods: list[HTTPMethod] = []
            for elt in kw.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    try:
                        methods.append(HTTPMethod(elt.value.upper()))
                    except ValueError:
                        pass
            return methods or [HTTPMethod.GET]

    return [HTTPMethod.GET]


def _path_from_decorator(decorator: ast.expr) -> str | None:
    """Extract the URL path from a route decorator."""
    if isinstance(decorator, ast.Call):
        # First positional arg is the path
        if decorator.args and isinstance(decorator.args[0], ast.Constant):
            val = decorator.args[0].value
            if isinstance(val, str):
                return val
    return None


def _is_flask_route_decorator(decorator: ast.expr, blueprint_names: set[str], app_names: set[str]) -> bool:
    """Return True if this looks like a Flask route decorator."""
    func = None
    if isinstance(decorator, ast.Call):
        func = decorator.func
    elif isinstance(decorator, ast.Attribute):
        func = decorator

    if not isinstance(func, ast.Attribute):
        return False

    owner_name: str | None = None
    if isinstance(func.value, ast.Name):
        owner_name = func.value.id

    method = func.attr
    is_route = method == "route" or method in _HTTP_METHODS_SET
    is_owned_by_flask_obj = owner_name in blueprint_names or owner_name in app_names
    return is_route and is_owned_by_flask_obj


_HTTP_METHODS_SET = {"get", "post", "put", "patch", "delete", "head", "options"}


def _collect_flask_app_names(tree: ast.Module) -> set[str]:
    """Find all variable names assigned a Flask() instance."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Call):
            func = node.value.func
            func_name = (
                func.id if isinstance(func, ast.Name)
                else func.attr if isinstance(func, ast.Attribute)
                else None
            )
            if func_name == "Flask":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
    return names or {"app", "application"}


def _collect_blueprints(tree: ast.Module) -> dict[str, str]:
    """Return a mapping of blueprint variable name → url_prefix."""
    blueprints: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        func_name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else None
        )
        if func_name != "Blueprint":
            continue
        prefix = ""
        for kw in node.value.keywords:
            if kw.arg == "url_prefix" and isinstance(kw.value, ast.Constant):
                prefix = kw.value.value
                break
        for target in node.targets:
            if isinstance(target, ast.Name):
                blueprints[target.id] = prefix
    return blueprints


def _collect_method_view_routes(tree: ast.Module, prefix: str = "") -> list[APIEndpoint]:
    """Extract routes from MethodView subclasses."""
    endpoints: list[APIEndpoint] = []

    # First find add_url_rule / register calls to figure out which path maps to which view
    # Simple heuristic: look for app.add_url_rule or bp.add_url_rule
    path_for_view: dict[str, str] = {}  # ClassName -> path
    for node in ast.walk(tree):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = call.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_url_rule":
            continue
        if len(call.args) < 2:
            continue
        path_arg = call.args[0]
        view_arg = call.args[1] if len(call.args) > 1 else None
        # view_func kwarg is also common
        for kw in call.keywords:
            if kw.arg == "view_func":
                view_arg = kw.value
                break
        if not isinstance(path_arg, ast.Constant) or not isinstance(path_arg.value, str):
            continue
        if view_arg is None:
            continue
        # view_arg is often ClassName.as_view('name')
        view_name: str | None = None
        if isinstance(view_arg, ast.Call) and isinstance(view_arg.func, ast.Attribute):
            if isinstance(view_arg.func.value, ast.Name):
                view_name = view_arg.func.value.id
        if view_name:
            path_for_view[view_name] = path_arg.value

    # Now find MethodView subclasses
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        is_method_view = any(
            (isinstance(b, ast.Name) and b.id == "MethodView")
            or (isinstance(b, ast.Attribute) and b.attr == "MethodView")
            for b in node.bases
        )
        if not is_method_view:
            continue

        path = path_for_view.get(node.name, f"/{node.name.lower()}")
        openapi_path, path_params = _flask_path_to_openapi(prefix + path)

        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            method_name = item.name.upper()
            if method_name not in {e.value for e in HTTPMethod}:
                continue
            try:
                method = HTTPMethod(method_name)
            except ValueError:
                continue
            doc = _extract_docstring(item)
            endpoints.append(APIEndpoint(
                path=openapi_path,
                method=method,
                summary=doc.splitlines()[0][:100] if doc else "",
                description=doc,
                parameters=list(path_params),
            ))

    return endpoints


def parse_flask_source(source: str, source_path: str) -> list[APIEndpoint]:
    """Parse Flask source and return discovered endpoints."""
    try:
        tree = ast.parse(source, filename=source_path)
    except SyntaxError as exc:
        logger.warning("Syntax error parsing %s: %s", source_path, exc)
        return []

    app_names = _collect_flask_app_names(tree)
    blueprints = _collect_blueprints(tree)
    all_known = app_names | set(blueprints.keys())

    endpoints: list[APIEndpoint] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not _is_flask_route_decorator(decorator, set(blueprints.keys()), app_names):
                continue

            route_path = _path_from_decorator(decorator)
            if route_path is None:
                continue

            # Determine prefix from blueprint
            owner_name: str | None = None
            func = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                owner_name = func.value.id

            prefix = blueprints.get(owner_name, "") if owner_name else ""
            full_path = prefix + route_path
            openapi_path, path_params = _flask_path_to_openapi(full_path)

            methods = _methods_from_node(decorator)
            doc = _extract_docstring(node)
            summary = doc.splitlines()[0][:100] if doc else ""

            for method in methods:
                key = f"{method.value} {openapi_path}"
                if key in seen:
                    continue
                seen.add(key)
                endpoints.append(APIEndpoint(
                    path=openapi_path,
                    method=method,
                    summary=summary,
                    description=doc,
                    parameters=list(path_params),
                ))

    # MethodView class-based routes
    for ep in _collect_method_view_routes(tree):
        key = f"{ep.method.value} {ep.path}"
        if key not in seen:
            seen.add(key)
            endpoints.append(ep)

    return endpoints


async def discover_flask(source_path: str) -> APISpec:
    """Discover Flask endpoints from a Python source file."""
    path = Path(source_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source file not found: {path}")

    source = path.read_text(encoding="utf-8")
    endpoints = parse_flask_source(source, str(path))

    return APISpec(
        name=path.stem,
        version="0.0.0",
        framework=FrameworkType.FLASK,
        endpoints=endpoints,
        source_path=str(path),
    )
