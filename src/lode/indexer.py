from __future__ import annotations

import ast
import hashlib
import os
import re
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph import resolve_graph
from .model import Edge, FileIndex, Node
from .storage import (
    connect,
    remove_missing_file_rows,
    replace_file_indexes_rows,
    upsert_repo,
)

SOURCE_EXTENSIONS = {
    ".py": "python",
    ".pyw": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".md": "markdown",
    ".mdx": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
}

AST_TRAVERSAL_LEAF_TYPES = (
    ast.expr_context,
    ast.operator,
    ast.unaryop,
    ast.cmpop,
    ast.boolop,
    ast.Constant,
    ast.Name,
    ast.arg,
    ast.alias,
)

EXCLUDED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "dist",
    "build",
    ".next",
    ".nuxt",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "coverage",
    ".local-repo-kg",
    ".lode",
    ".kg",
    "bench-results",
    "droid-wiki",
}

PARALLEL_PARSE_THRESHOLD = 8
MAX_PARSE_WORKERS = 8
ParseJob = tuple[str, str, str, float, int, str]


@dataclass(slots=True)
class IndexStats:
    repo_id: str
    root: str
    scanned: int = 0
    indexed: int = 0
    skipped_unchanged: int = 0
    removed: int = 0
    resolved_imports: int = 0
    resolved_calls: int = 0
    resolved_extends: int = 0
    nodes: int = 0
    edges: int = 0


def index_repo(repo_path: Path, db_path: Path | None = None) -> IndexStats:
    root = repo_path.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Repository path does not exist or is not a directory: {root}")
    conn = connect(db_path)
    try:
        repo_id = upsert_repo(conn, root)
        stats = IndexStats(repo_id=repo_id, root=str(root))
        live_paths: set[str] = set()
        parse_jobs: list[ParseJob] = []
        skipped_updates: list[tuple[float, float, str, str]] = []
        previous_files = {
            row["path"]: {
                "content_hash": row["content_hash"],
                "size": row["size"],
                "mtime": row["mtime"],
            }
            for row in conn.execute(
                "SELECT path, content_hash, size, mtime FROM files WHERE repo_id = ?", (repo_id,)
            )
        }
        for path in iter_source_files(root):
            rel = path.relative_to(root).as_posix()
            stats.scanned += 1
            try:
                stat = path.stat()
            except OSError:
                continue
            live_paths.add(rel)
            prev = previous_files.get(rel)
            try:
                raw = path.read_bytes()
            except OSError:
                live_paths.discard(rel)
                continue
            digest = hashlib.sha1(raw).hexdigest()
            if prev and prev["content_hash"] == digest:
                stats.skipped_unchanged += 1
                skipped_updates.append((stat.st_mtime, time.time(), repo_id, rel))
                continue
            text = raw.decode("utf-8", errors="replace")
            parse_jobs.append((str(root), str(path), digest, stat.st_mtime, stat.st_size, text))
        if should_parse_in_parallel(parse_jobs):
            conn.commit()
            conn.close()
            changed_files = parse_file_jobs(parse_jobs)
            conn = connect(db_path)
        else:
            changed_files = parse_file_jobs(parse_jobs)
        stats.indexed = len(changed_files)
        stats.nodes = sum(len(file_index.nodes) for file_index in changed_files)
        stats.edges = sum(len(file_index.edges) for file_index in changed_files)
        with conn:
            if skipped_updates:
                conn.executemany(
                    "UPDATE files SET mtime = ?, indexed_at = ? WHERE repo_id = ? AND path = ?",
                    skipped_updates,
                )
            replace_file_indexes_rows(conn, repo_id, changed_files)
            stats.removed = remove_missing_file_rows(conn, repo_id, live_paths)
        if stats.indexed > 0 or stats.removed > 0:
            resolved = resolve_graph(conn, repo_id)
            stats.resolved_imports = resolved.get("imports", 0)
            stats.resolved_calls = resolved.get("calls", 0)
            stats.resolved_extends = resolved.get("extends", 0)
        return stats
    finally:
        conn.close()


def should_parse_in_parallel(parse_jobs: list[ParseJob]) -> bool:
    return len(parse_jobs) >= PARALLEL_PARSE_THRESHOLD and (os.cpu_count() or 1) > 1


def parse_file_jobs(parse_jobs: list[ParseJob]) -> list[FileIndex]:
    if not parse_jobs:
        return []
    if not should_parse_in_parallel(parse_jobs):
        return [_parse_file_job(job) for job in parse_jobs]
    worker_count = min(MAX_PARSE_WORKERS, len(parse_jobs), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        return list(executor.map(_parse_file_job, parse_jobs, chunksize=1))


def _parse_file_job(job: ParseJob) -> FileIndex:
    root, path, digest, mtime, size, text = job
    return parse_file(Path(root), Path(path), digest, mtime, size, text)


def iter_source_files(root: Path):
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        current = Path(current_root)
        for filename in files:
            path = current / filename
            if path.suffix.lower() not in SOURCE_EXTENSIONS:
                continue
            if is_probably_generated(path):
                continue
            yield path


def is_probably_generated(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith((".min.js", ".bundle.js")):
        return True
    if "generated" in name or name.endswith(".lock"):
        return True
    try:
        return path.stat().st_size > 1_000_000
    except OSError:
        return True


def parse_file(
    root: Path,
    path: Path,
    digest: str | None = None,
    mtime: float = 0.0,
    size: int | None = None,
    text: str | None = None,
) -> FileIndex:
    rel = path.relative_to(root).as_posix()
    language = SOURCE_EXTENSIONS.get(path.suffix.lower(), "text")
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
    content_hash = digest or hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()
    file_node = make_node(
        "File",
        Path(rel).name,
        rel,
        rel,
        1,
        max(1, text.count("\n") + 1),
        content_hash,
        doc="",
    )
    nodes = [file_node]
    edges: list[Edge] = []
    if language == "python":
        extra_nodes, extra_edges, import_bindings = parse_python(rel, text, file_node.id)
    elif language in {"typescript", "javascript"}:
        extra_nodes, extra_edges, import_bindings = parse_ts_js(rel, text, file_node.id, language)
    elif language == "markdown":
        extra_nodes, extra_edges = parse_markdown(rel, text, file_node.id)
        import_bindings = []
    else:
        extra_nodes, extra_edges = parse_config(rel, text, file_node.id, language)
        import_bindings = []
    nodes.extend(extra_nodes)
    edges.extend(extra_edges)
    if import_bindings:
        file_node.extra["imports"] = import_bindings
    file_size = size if size is not None else path.stat().st_size
    return FileIndex(
        path=rel,
        abspath=str(path),
        language=language,
        size=file_size,
        mtime=mtime,
        content_hash=content_hash,
        generated=False,
        nodes=nodes,
        edges=edges,
    )


def make_node(
    kind: str,
    name: str,
    qname: str,
    path: str,
    start_line: int,
    end_line: int,
    content: str,
    signature: str = "",
    doc: str = "",
    confidence: str = "strong",
    extra: dict | None = None,
) -> Node:
    key = f"{kind}\0{qname}\0{path}\0{start_line}"
    node_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:24]
    content_hash = hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()
    return Node(
        id=node_id,
        kind=kind,
        name=name,
        qname=qname,
        path=path,
        start_line=start_line,
        end_line=end_line,
        signature=signature,
        doc=doc,
        confidence=confidence,
        content_hash=content_hash,
        extra=extra or {},
    )


def module_qname(rel: str) -> str:
    path = Path(rel)
    without_suffix = path.with_suffix("").as_posix()
    return without_suffix.replace("/", ".")


def _split_import_clause(clause: str) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    if not clause:
        return parts
    default_match = re.match(r"^(\w+)", clause)
    if default_match:
        parts.append({"name": "default", "alias": default_match.group(1)})
    namespace_match = re.search(r"\*\s+as\s+(\w+)", clause)
    if namespace_match:
        parts.append({"name": "*", "alias": namespace_match.group(1)})
    named_match = re.search(r"\{([^}]+)\}", clause)
    if named_match:
        for item in named_match.group(1).split(","):
            item = item.strip()
            if not item:
                continue
            if " as " in item:
                orig, alias = item.split(" as ", 1)
                parts.append({"name": orig.strip(), "alias": alias.strip()})
            else:
                parts.append({"name": item, "alias": None})
    if not parts and not default_match and not namespace_match and not named_match:
        parts.append({"name": None, "alias": None})
    return parts


def parse_python(
    rel: str, text: str, file_node_id: str
) -> tuple[list[Node], list[Edge], list[dict[str, Any]]]:
    nodes: list[Node] = []
    edges: list[Edge] = []
    import_bindings: list[dict[str, Any]] = []
    external_cache: dict[tuple[str, str], Node] = {}

    def cached_external_node(kind: str, name: str) -> Node:
        key = (kind, name)
        node = external_cache.get(key)
        if node is None:
            node = external_node(kind, name, rel)
            external_cache[key] = node
        return node

    module = module_qname(rel)
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        node = make_node(
            "ParseError",
            "SyntaxError",
            f"{module}.SyntaxError",
            rel,
            exc.lineno or 1,
            exc.lineno or 1,
            text,
            doc=str(exc),
            confidence="exact",
        )
        return [node], [Edge(file_node_id, node.id, "HAS_PARSE_ERROR", "exact")], []

    lines = text.splitlines()
    symbol_by_name: dict[str, str] = {}
    for item in tree.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node = python_function_node(module, rel, item, lines)
            nodes.append(node)
            symbol_by_name[item.name] = node.id
            edges.append(Edge(file_node_id, node.id, "DEFINES", "exact"))
            route_nodes, route_edges = route_facts_for_python(item, node, rel, file_node_id)
            nodes.extend(route_nodes)
            edges.extend(route_edges)
        elif isinstance(item, ast.ClassDef):
            class_node = python_class_node(module, rel, item, lines)
            nodes.append(class_node)
            symbol_by_name[item.name] = class_node.id
            edges.append(Edge(file_node_id, class_node.id, "DEFINES", "exact"))
            for base in item.bases:
                base_name = dotted_name(base)
                if base_name:
                    base_node = cached_external_node("ExternalSymbol", base_name)
                    nodes.append(base_node)
                    edges.append(
                        Edge(
                            class_node.id,
                            base_node.id,
                            "EXTENDS",
                            "heuristic",
                            base_name,
                        )
                    )
            for child in item.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method = python_function_node(
                        f"{module}.{item.name}", rel, child, lines, kind="Method"
                    )
                    nodes.append(method)
                    edges.append(Edge(class_node.id, method.id, "CONTAINS", "exact"))
                    edges.append(Edge(file_node_id, method.id, "DEFINES", "exact"))
                    route_nodes, route_edges = route_facts_for_python(
                        child, method, rel, file_node_id
                    )
                    nodes.extend(route_nodes)
                    edges.extend(route_edges)

    call_nodes: list[ast.Call] = []
    for import_node in iter_imports_and_calls(tree):
        if isinstance(import_node, ast.Import):
            for alias in import_node.names:
                import_module = alias.name
                import_bindings.append(
                    {"module": import_module, "name": None, "alias": alias.asname}
                )
                dep = cached_external_node("ExternalDependency", import_module)
                nodes.append(dep)
                edges.append(Edge(file_node_id, dep.id, "IMPORTS", "strong", import_module))
        elif isinstance(import_node, ast.ImportFrom):
            module = import_node.module or ""
            if import_node.level:
                module = "." * import_node.level + module
            for alias in import_node.names:
                if alias.name == "*":
                    import_bindings.append({"module": module, "name": "*", "alias": None})
                elif alias.asname:
                    import_bindings.append(
                        {"module": module, "name": alias.name, "alias": alias.asname}
                    )
                else:
                    import_bindings.append({"module": module, "name": alias.name, "alias": None})
            if module:
                dep = cached_external_node("ExternalDependency", module)
                nodes.append(dep)
                edges.append(Edge(file_node_id, dep.id, "IMPORTS", "strong", module))
        elif isinstance(import_node, ast.Call):
            call_nodes.append(import_node)

    function_stack = build_python_function_ranges(nodes)
    for call in call_nodes:
        call_name = dotted_name(call.func)
        if not call_name:
            continue
        caller_id = enclosing_node_for_line(function_stack, getattr(call, "lineno", 0))
        if not caller_id:
            continue
        target_id = symbol_by_name.get(call_name) if "." not in call_name else None
        if target_id:
            edges.append(Edge(caller_id, target_id, "CALLS", "strong", call_name))
        else:
            target = cached_external_node("ExternalSymbol", call_name)
            nodes.append(target)
            edges.append(Edge(caller_id, target.id, "CALLS", "heuristic", call_name))
    return nodes, edges, import_bindings


def iter_imports_and_calls(tree: ast.AST) -> Iterator[ast.Import | ast.ImportFrom | ast.Call]:
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
            yield node
        children = [
            child
            for child in ast.iter_child_nodes(node)
            if not isinstance(child, AST_TRAVERSAL_LEAF_TYPES)
        ]
        stack.extend(reversed(children))


def python_function_node(
    prefix: str,
    rel: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: list[str],
    kind: str = "Function",
) -> Node:
    args = [arg.arg for arg in node.args.args]
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    signature = f"{node.name}({', '.join(args)})"
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    doc = ast.get_docstring(node) or ""
    body = slice_lines(lines, start, end)
    return make_node(
        kind,
        node.name,
        f"{prefix}.{node.name}",
        rel,
        start,
        end,
        body,
        signature,
        doc,
        "exact",
    )


def python_class_node(module: str, rel: str, node: ast.ClassDef, lines: list[str]) -> Node:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    bases = [dotted_name(base) for base in node.bases]
    bases = [base for base in bases if base]
    signature = f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
    doc = ast.get_docstring(node) or ""
    body = slice_lines(lines, start, end)
    return make_node(
        "Class",
        node.name,
        f"{module}.{node.name}",
        rel,
        start,
        end,
        body,
        signature,
        doc,
        "exact",
    )


def route_facts_for_python(
    item: ast.FunctionDef | ast.AsyncFunctionDef,
    handler: Node,
    rel: str,
    file_node_id: str,
) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []
    for decorator in item.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        method_name = dotted_name(decorator.func)
        if not method_name:
            continue
        method = method_name.split(".")[-1].upper()
        if method not in {
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
            "HEAD",
            "ROUTE",
        }:
            continue
        route_path = None
        if (
            decorator.args
            and isinstance(decorator.args[0], ast.Constant)
            and isinstance(decorator.args[0].value, str)
        ):
            route_path = decorator.args[0].value
        if route_path:
            route = make_node(
                "Route",
                f"{method} {route_path}",
                f"route:{method} {route_path}",
                rel,
                item.lineno,
                item.lineno,
                route_path,
                signature=f"{method} {route_path}",
                confidence="strong",
            )
            nodes.append(route)
            edges.append(Edge(file_node_id, route.id, "DEFINES", "strong", method_name))
            edges.append(Edge(route.id, handler.id, "HANDLES", "strong", method_name))
            edges.append(Edge(route.id, handler.id, "CALLS", "heuristic", method_name))
            edges.append(Edge(handler.id, route.id, "EXPOSES", "strong", method_name))
    return nodes, edges


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = dotted_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    if isinstance(node, ast.Subscript):
        return dotted_name(node.value)
    return ""


def build_python_function_ranges(nodes: list[Node]) -> list[tuple[int, int, str]]:
    ranges = [
        (node.start_line, node.end_line, node.id)
        for node in nodes
        if node.kind in {"Function", "Method"}
    ]
    return sorted(ranges, key=lambda item: (item[1] - item[0], item[0]))


def enclosing_node_for_line(ranges: list[tuple[int, int, str]], line: int) -> str | None:
    for start, end, node_id in ranges:
        if start <= line <= end:
            return node_id
    return None


def parse_ts_js(
    rel: str, text: str, file_node_id: str, language: str
) -> tuple[list[Node], list[Edge], list[dict[str, Any]]]:
    nodes: list[Node] = []
    edges: list[Edge] = []
    import_bindings: list[dict[str, Any]] = []
    module = module_qname(rel)
    lines = text.splitlines()

    for match in re.finditer(r"import\s+(.*?)\s+from\s+(['\"])([^'\"]+)\2", text):
        clause = match.group(1).strip()
        mod = match.group(3)
        for part in _split_import_clause(clause):
            import_bindings.append({"module": mod, **part})
        dep = external_node("ExternalDependency", mod, rel)
        nodes.append(dep)
        edges.append(Edge(file_node_id, dep.id, "IMPORTS", "strong", mod))
    for match in re.finditer(r"require\(\s*(['\"])([^'\"]+)\1\s*\)", text):
        mod = match.group(2)
        import_bindings.append({"module": mod, "name": None, "alias": None})
        dep = external_node("ExternalDependency", mod, rel)
        nodes.append(dep)
        edges.append(Edge(file_node_id, dep.id, "IMPORTS", "strong", mod))

    symbol_by_name: dict[str, str] = {}
    patterns = [
        ("Class", re.compile(r"(?:export\s+)?class\s+([A-Za-z_$][\w$]*)")),
        (
            "Function",
            re.compile(r"(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)"),
        ),
        (
            "Function",
            re.compile(
                r"(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>"
            ),
        ),
        (
            "Function",
            re.compile(
                r"(?:export\s+)?const\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?([A-Za-z_$][\w$]*)\s*=>"
            ),
        ),
    ]
    for line_no, line in enumerate(lines, start=1):
        for kind, pattern in patterns:
            symbol_match = pattern.search(line)
            if not symbol_match:
                continue
            name = symbol_match.group(1)
            end = find_block_end(lines, line_no)
            signature = line.strip()[:240]
            body = "\n".join(lines[line_no - 1 : end])
            node = make_node(
                kind,
                name,
                f"{module}.{name}",
                rel,
                line_no,
                end,
                body,
                signature,
                "",
                "heuristic",
                {"language": language},
            )
            nodes.append(node)
            symbol_by_name[name] = node.id
            edges.append(Edge(file_node_id, node.id, "DEFINES", "heuristic"))

    for line_no, line in enumerate(lines, start=1):
        route_match = re.search(
            r"\b(?:app|router)\.(get|post|put|patch|delete|options|head)\s*\(\s*(['\"])([^'\"]+)\2\s*,\s*([A-Za-z_$][\w$]*)",
            line,
        )
        if route_match:
            method = route_match.group(1).upper()
            route_path = route_match.group(3)
            handler = route_match.group(4)
            route = make_node(
                "Route",
                f"{method} {route_path}",
                f"route:{method} {route_path}",
                rel,
                line_no,
                line_no,
                line,
                signature=f"{method} {route_path}",
                confidence="heuristic",
            )
            nodes.append(route)
            edges.append(Edge(file_node_id, route.id, "DEFINES", "heuristic"))
            target_id = symbol_by_name.get(handler)
            if target_id:
                edges.append(Edge(route.id, target_id, "HANDLES", "heuristic", handler))

    function_ranges = [
        (node.start_line, node.end_line, node.id) for node in nodes if node.kind == "Function"
    ]
    for line_no, line in enumerate(lines, start=1):
        caller_id = enclosing_node_for_line(function_ranges, line_no)
        if not caller_id:
            continue
        for call_name in re.findall(r"\b([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(", line):
            first_segment = call_name.split(".", 1)[0]
            if first_segment in {
                "if",
                "for",
                "while",
                "switch",
                "return",
                "function",
            }:
                continue
            target_id = symbol_by_name.get(call_name) if "." not in call_name else None
            if target_id:
                if target_id != caller_id:
                    edges.append(Edge(caller_id, target_id, "CALLS", "heuristic", call_name))
                continue
            target = external_node("ExternalSymbol", call_name, rel)
            nodes.append(target)
            edges.append(Edge(caller_id, target.id, "CALLS", "heuristic", call_name))
    return nodes, edges, import_bindings


def find_block_end(lines: list[str], start_line: int) -> int:
    depth = 0
    seen_open = False
    for idx in range(start_line - 1, min(len(lines), start_line + 400)):
        line = lines[idx]
        depth += line.count("{")
        if "{" in line:
            seen_open = True
        depth -= line.count("}")
        if seen_open and depth <= 0:
            return idx + 1
    return start_line


def parse_markdown(rel: str, text: str, file_node_id: str) -> tuple[list[Node], list[Edge]]:
    nodes: list[Node] = []
    edges: list[Edge] = []
    headings = list(re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, re.MULTILINE))
    lines = text.splitlines()
    for idx, match in enumerate(headings):
        title = match.group(2).strip()
        start = text[: match.start()].count("\n") + 1
        end = headings[idx + 1].start() if idx + 1 < len(headings) else len(text)
        end_line = text[:end].count("\n") + 1
        body = "\n".join(lines[start - 1 : end_line])
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        node = make_node(
            "DocSection",
            title,
            f"{rel}#{slug}",
            rel,
            start,
            end_line,
            body,
            signature="#" * len(match.group(1)) + " " + title,
            doc=body[:1000],
            confidence="exact",
        )
        nodes.append(node)
        edges.append(Edge(file_node_id, node.id, "CONTAINS", "exact"))
    return nodes, edges


def parse_config(
    rel: str, text: str, file_node_id: str, language: str
) -> tuple[list[Node], list[Edge]]:
    node = make_node(
        "ConfigFile",
        Path(rel).name,
        rel,
        rel,
        1,
        max(1, text.count("\n") + 1),
        text[:4000],
        signature=language,
        confidence="exact",
    )
    return [node], [Edge(file_node_id, node.id, "CONTAINS", "exact")]


def external_node(kind: str, name: str, rel: str) -> Node:
    return make_node(kind, name, f"external:{name}", rel, 1, 1, name, confidence="heuristic")


def slice_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[max(0, start - 1) : max(start, end)])
