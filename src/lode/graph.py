"""Graph resolution and blast-radius traversal over the SQLite index.

This module turns per-file parse facts into a connected codebase graph:

- file -> file IMPORTS edges (Python absolute/relative imports, TS/JS paths)
- import-aware CALLS resolution (disambiguates same-named symbols)
- EXTENDS resolution to local classes (subclass impact)
- bounded breadth-first blast radius with distances and confidence
"""

from __future__ import annotations

import json
import posixpath
import sqlite3
from collections import deque
from typing import Any

from .storage import find_symbol, get_neighbors, get_node, search_nodes

CONFIDENCE_RANK = {"exact": 0, "strong": 1, "resolved": 2, "heuristic": 3}
RANK_CONFIDENCE = {rank: name for name, rank in CONFIDENCE_RANK.items()}

TRAVERSAL_EDGE_KINDS = ("CALLS", "EXTENDS", "HANDLES", "CONTAINS", "IMPORTS")
EXPANDABLE_NODE_KINDS = {"File", "Function", "Method", "Class", "Route"}
INTERNAL_NODE_KINDS = {"File", "Function", "Method", "Class", "Route", "DocSection"}
CALLABLE_KINDS = ("Function", "Class", "Method")

JS_SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")

_UPSTREAM_SQL = """
SELECT e.kind AS edge_kind, e.confidence AS edge_confidence, e.detail AS edge_detail,
       n.id, n.kind, n.name, n.qname, n.path, n.start_line, n.end_line
FROM edges e
JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.src
WHERE e.repo_id = ? AND e.dst = ? AND e.kind IN ('CALLS','EXTENDS','HANDLES','CONTAINS','IMPORTS')
ORDER BY
  CASE e.confidence WHEN 'exact' THEN 0 WHEN 'strong' THEN 1 WHEN 'resolved' THEN 2 ELSE 3 END,
  n.path, n.start_line
LIMIT ?
"""

_DOWNSTREAM_SQL = """
SELECT e.kind AS edge_kind, e.confidence AS edge_confidence, e.detail AS edge_detail,
       n.id, n.kind, n.name, n.qname, n.path, n.start_line, n.end_line
FROM edges e
JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
WHERE e.repo_id = ? AND e.src = ? AND e.kind IN ('CALLS','EXTENDS','HANDLES','CONTAINS','IMPORTS')
ORDER BY
  CASE e.confidence WHEN 'exact' THEN 0 WHEN 'strong' THEN 1 WHEN 'resolved' THEN 2 ELSE 3 END,
  n.path, n.start_line
LIMIT ?
"""


def resolve_graph(conn: sqlite3.Connection, repo_id: str) -> dict[str, int]:
    """Rebuild all confidence='resolved' edges for a repo from parse facts."""
    file_languages = {
        row["path"]: row["language"]
        for row in conn.execute(
            "SELECT path, language FROM files WHERE repo_id = ?", (repo_id,)
        )
    }
    file_paths = set(file_languages)
    file_nodes: dict[str, str] = {}
    imports_by_path: dict[str, list[dict[str, Any]]] = {}
    for row in conn.execute(
        "SELECT id, path, extra_json FROM nodes WHERE repo_id = ? AND kind = 'File'",
        (repo_id,),
    ):
        file_nodes[row["path"]] = row["id"]
        try:
            extra = json.loads(row["extra_json"] or "{}")
        except json.JSONDecodeError:
            extra = {}
        bindings = extra.get("imports")
        if isinstance(bindings, list):
            imports_by_path[row["path"]] = [
                binding for binding in bindings if isinstance(binding, dict)
            ]

    def_rows = list(
        conn.execute(
            """
            SELECT id, kind, name, qname, path
            FROM nodes
            WHERE repo_id = ? AND kind IN ('Function', 'Method', 'Class')
            """,
            (repo_id,),
        )
    )
    by_qname = {row["qname"]: row for row in def_rows}
    by_name: dict[str, list[sqlite3.Row]] = {}
    defs_by_file: dict[str, dict[str, list[sqlite3.Row]]] = {}
    methods_by_class: dict[tuple[str, str, str], list[sqlite3.Row]] = {}
    for row in def_rows:
        by_name.setdefault(row["name"], []).append(row)
        defs_by_file.setdefault(row["path"], {}).setdefault(row["name"], []).append(row)
        if row["kind"] == "Method" and "." in row["qname"]:
            class_qname = row["qname"].rsplit(".", 1)[0]
            methods_by_class.setdefault(
                (row["path"], class_qname, row["name"]), []
            ).append(row)

    module_cache: dict[tuple[str, str], str | None] = {}

    def resolve_module(module: str, importer: str) -> str | None:
        key = (module, importer)
        if key not in module_cache:
            language = file_languages.get(importer, "")
            if language == "python":
                module_cache[key] = resolve_python_module(module, importer, file_paths)
            elif language in {"typescript", "javascript"}:
                module_cache[key] = resolve_js_module(module, importer, file_paths)
            else:
                module_cache[key] = None
        return module_cache[key]

    def pick_def(
        path: str | None, name: str | None, prefer: tuple[str, ...]
    ) -> sqlite3.Row | None:
        if not path or not name:
            return None
        rows = defs_by_file.get(path, {}).get(name, [])
        preferred = [row for row in rows if row["kind"] in prefer]
        pool = preferred or rows
        return pool[0] if len(pool) == 1 else None

    def resolve_via_imports(
        caller_path: str, dotted: str, prefer: tuple[str, ...]
    ) -> sqlite3.Row | None:
        bindings = imports_by_path.get(caller_path)
        if not bindings:
            return None
        alias_map: dict[str, list[tuple[str, str | None]]] = {}
        star_modules: list[str] = []

        def add_alias(alias: str | None, module: str, name: str | None) -> None:
            alias = (alias or "").strip()
            if alias:
                alias_map.setdefault(alias, []).append((module, name))

        for binding in bindings:
            module = str(binding.get("module") or "").strip()
            if not module:
                continue
            raw_name = binding.get("name")
            name = str(raw_name).strip() if raw_name is not None else None
            if name == "*":
                star_modules.append(module)
                continue
            alias = str(binding.get("alias") or "").strip()
            if alias:
                add_alias(alias, module, name)
            elif name not in (None, "", "default"):
                add_alias(name, module, name)
            else:
                add_alias(module, module, None)

        segments = dotted.split(".")
        for index in range(len(segments), 0, -1):
            prefix = ".".join(segments[:index])
            if prefix not in alias_map:
                continue
            remainder = segments[index:]
            found: dict[str, sqlite3.Row] = {}
            for module, name in alias_map[prefix]:
                row: sqlite3.Row | None = None
                if name in (None, ""):
                    if remainder:
                        row = pick_def(
                            resolve_module(module, caller_path), remainder[0], prefer
                        )
                elif name == "default":
                    symbol = remainder[0] if remainder else prefix
                    row = pick_def(resolve_module(module, caller_path), symbol, prefer)
                else:
                    if remainder:
                        submodule = join_module_name(module, name)
                        row = pick_def(
                            resolve_module(submodule, caller_path),
                            remainder[0],
                            prefer,
                        )
                        if row is None:
                            row = pick_def(
                                resolve_module(module, caller_path),
                                remainder[0],
                                prefer,
                            )
                    else:
                        row = pick_def(resolve_module(module, caller_path), name, prefer)
                if row is not None:
                    found[row["id"]] = row
            if len(found) == 1:
                return next(iter(found.values()))
            if found:
                return None

        if len(segments) == 1 and star_modules:
            found: dict[str, sqlite3.Row] = {}
            for module in star_modules:
                row = pick_def(resolve_module(module, caller_path), segments[0], prefer)
                if row is not None:
                    found[row["id"]] = row
            if len(found) == 1:
                return next(iter(found.values()))
        return None

    def resolve_self_or_class_method(
        caller: sqlite3.Row | None, dotted: str, prefer: tuple[str, ...]
    ) -> sqlite3.Row | None:
        if caller is None:
            return None
        segments = dotted.split(".")
        if len(segments) != 2 or segments[0] not in {"self", "cls", "this"}:
            return None
        if caller["caller_kind"] != "Method" or "." not in caller["caller_qname"]:
            return None
        class_qname = caller["caller_qname"].rsplit(".", 1)[0]
        rows = methods_by_class.get(
            (caller["caller_path"], class_qname, segments[1]), []
        )
        pool = [row for row in rows if row["kind"] in prefer] or rows
        return pool[0] if len(pool) == 1 else None

    def resolve_target(
        caller_path: str,
        dotted: str,
        prefer: tuple[str, ...],
        caller: sqlite3.Row | None = None,
    ) -> sqlite3.Row | None:
        dotted = (dotted or "").strip()
        if not dotted:
            return None
        exact = by_qname.get(dotted)
        if exact is not None:
            return exact
        row = resolve_self_or_class_method(caller, dotted, prefer)
        if row is not None:
            return row
        if dotted.split(".")[0] in {"self", "cls", "this"}:
            return None
        row = resolve_via_imports(caller_path, dotted, prefer)
        if row is not None:
            return row
        if "." in dotted:
            suffix = "." + dotted
            matches = [r for r in def_rows if r["qname"].endswith(suffix)]
            if len(matches) == 1:
                return matches[0]
        simple = dotted.rsplit(".", 1)[-1]
        matches = by_name.get(simple, [])
        pool = [r for r in matches if r["kind"] in prefer] or matches
        if len(pool) == 1:
            return pool[0]
        return None

    counts = {"imports": 0, "calls": 0, "extends": 0}
    with conn:
        conn.execute(
            "DELETE FROM edges WHERE repo_id = ? AND confidence = 'resolved'",
            (repo_id,),
        )

        for path, bindings in imports_by_path.items():
            src_id = file_nodes.get(path)
            if not src_id:
                continue
            for binding in bindings:
                module = str(binding.get("module") or "").strip()
                if not module:
                    continue
                name = binding.get("name")
                detail = module
                target_path = None
                if name and name not in ("*", "default"):
                    joined = join_module_name(module, str(name))
                    target_path = resolve_module(joined, path)
                    if target_path:
                        detail = joined
                if target_path is None:
                    target_path = resolve_module(module, path)
                if not target_path or target_path == path:
                    continue
                dst_id = file_nodes.get(target_path)
                if not dst_id:
                    continue
                result = conn.execute(
                    """
                    INSERT OR IGNORE INTO edges(
                        repo_id, owner_path, src, dst, kind, confidence, detail
                    ) VALUES (?, ?, ?, ?, 'IMPORTS', 'resolved', ?)
                    """,
                    (repo_id, path, src_id, dst_id, detail),
                )
                counts["imports"] += max(result.rowcount, 0)

        unresolved_calls = list(
            conn.execute(
                """
                SELECT e.owner_path, e.src, e.detail, target.name AS target_name,
                       caller.kind AS caller_kind,
                       caller.qname AS caller_qname,
                       caller.path AS caller_path
                FROM edges e
                JOIN nodes target ON target.repo_id = e.repo_id AND target.id = e.dst
                JOIN nodes caller ON caller.repo_id = e.repo_id AND caller.id = e.src
                WHERE e.repo_id = ?
                  AND e.kind = 'CALLS'
                  AND target.kind = 'ExternalSymbol'
                """,
                (repo_id,),
            )
        )
        for edge in unresolved_calls:
            call_name = (edge["detail"] or edge["target_name"] or "").strip()
            row = resolve_target(
                edge["owner_path"], call_name, CALLABLE_KINDS, caller=edge
            )
            if row is None or row["id"] == edge["src"]:
                continue
            result = conn.execute(
                """
                INSERT OR IGNORE INTO edges(
                    repo_id, owner_path, src, dst, kind, confidence, detail
                ) VALUES (?, ?, ?, ?, 'CALLS', 'resolved', ?)
                """,
                (repo_id, edge["owner_path"], edge["src"], row["id"], call_name),
            )
            counts["calls"] += max(result.rowcount, 0)

        unresolved_extends = list(
            conn.execute(
                """
                SELECT e.owner_path, e.src, e.detail, target.name AS target_name
                FROM edges e
                JOIN nodes target ON target.repo_id = e.repo_id AND target.id = e.dst
                WHERE e.repo_id = ?
                  AND e.kind = 'EXTENDS'
                  AND target.kind = 'ExternalSymbol'
                """,
                (repo_id,),
            )
        )
        for edge in unresolved_extends:
            base_name = (edge["detail"] or edge["target_name"] or "").strip()
            row = resolve_target(edge["owner_path"], base_name, ("Class",))
            if row is None or row["id"] == edge["src"]:
                continue
            result = conn.execute(
                """
                INSERT OR IGNORE INTO edges(
                    repo_id, owner_path, src, dst, kind, confidence, detail
                ) VALUES (?, ?, ?, ?, 'EXTENDS', 'resolved', ?)
                """,
                (repo_id, edge["owner_path"], edge["src"], row["id"], base_name),
            )
            counts["extends"] += max(result.rowcount, 0)
    return counts


def join_module_name(module: str, name: str) -> str:
    if not module:
        return name
    if module.endswith("."):
        return module + name
    return f"{module}.{name}"


def resolve_python_module(
    module: str, importer: str, file_paths: set[str]
) -> str | None:
    module = module.strip()
    if not module:
        return None
    if module.startswith("."):
        level = len(module) - len(module.lstrip("."))
        rest = module[level:]
        parts = importer.split("/")[:-1]
        ups = level - 1
        if ups > len(parts):
            return None
        base = parts[: len(parts) - ups] if ups else parts
        segments = [segment for segment in rest.split(".") if segment]
        rel = "/".join([*base, *segments])
        if rel:
            candidates = (f"{rel}.py", f"{rel}/__init__.py")
        else:
            candidates = ("__init__.py",)
        for candidate in candidates:
            if candidate in file_paths:
                return candidate
        return None
    rel = module.replace(".", "/")
    candidates = (f"{rel}.py", f"{rel}/__init__.py")
    for candidate in candidates:
        if candidate in file_paths:
            return candidate
    for candidate in candidates:
        matches = [path for path in file_paths if path.endswith("/" + candidate)]
        if len(matches) == 1:
            return matches[0]
    return None


def resolve_js_module(module: str, importer: str, file_paths: set[str]) -> str | None:
    module = module.strip()
    if not module.startswith("."):
        return None
    base = posixpath.dirname(importer)
    target = posixpath.normpath(posixpath.join(base, module))
    if target.startswith(".."):
        return None
    candidates = [target]
    stem, ext = posixpath.splitext(target)
    if ext in {".js", ".mjs", ".cjs"}:
        candidates.extend([stem + ".ts", stem + ".tsx"])
    candidates.extend(target + suffix for suffix in JS_SOURCE_SUFFIXES)
    candidates.extend(
        posixpath.join(target, f"index{suffix}") for suffix in JS_SOURCE_SUFFIXES
    )
    for candidate in candidates:
        if candidate in file_paths:
            return candidate
    return None


def weakest_confidence(left: str | None, right: str | None) -> str:
    rank = max(
        CONFIDENCE_RANK.get(left or "", 3), CONFIDENCE_RANK.get(right or "", 3)
    )
    return RANK_CONFIDENCE[rank]


def edge_scope(edge_kind: str) -> str:
    return "conservative_file" if edge_kind == "IMPORTS" else "precise_symbol"


def compact_radius_node(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "kind": row["kind"],
        "name": row["name"],
        "qname": row["qname"],
        "path": row["path"],
        "lines": [row["start_line"], row["end_line"]],
    }


def blast_radius(
    conn: sqlite3.Connection,
    node_id: str,
    depth: int | None = None,
    max_nodes: int = 1000,
    direction: str = "both",
) -> dict[str, Any]:
    """Bounded BFS over dependency edges around a node.

    upstream = nodes that depend on the target (break when it changes),
    downstream = nodes the target depends on.

    By default traversal is complete for the reachable graph and bounded by
    max_nodes. Pass depth to intentionally limit noisy or very large graphs.
    """
    if direction not in {"up", "down", "both"}:
        raise ValueError("direction must be 'up', 'down', or 'both'")
    target = get_node(conn, node_id)
    if not target:
        return {}
    repo_id = str(target["repo_id"])
    depth_limit = None if depth is None or int(depth) <= 0 else int(depth)
    max_nodes = max(1, min(int(max_nodes), 10000))

    seeds_up: dict[str, str] = {node_id: str(target.get("qname") or "")}
    seeds_down: dict[str, str] = dict(seeds_up)
    if target.get("kind") == "File":
        for row in conn.execute(
            "SELECT dst FROM edges WHERE repo_id = ? AND src = ? AND kind = 'DEFINES'",
            (repo_id, node_id),
        ):
            seeds_up[row["dst"]] = str(target.get("qname") or "")
            seeds_down[row["dst"]] = str(target.get("qname") or "")
    elif target.get("path"):
        file_row = conn.execute(
            "SELECT id FROM nodes WHERE repo_id = ? AND kind = 'File' AND path = ? LIMIT 1",
            (repo_id, target["path"]),
        ).fetchone()
        if file_row:
            seeds_up[file_row["id"]] = str(target.get("qname") or "")

    upstream: list[dict[str, Any]] = []
    downstream: list[dict[str, Any]] = []
    up_truncated = down_truncated = False
    if direction in {"up", "both"}:
        upstream, up_truncated = _walk(
            conn, repo_id, seeds_up, _UPSTREAM_SQL, "up", depth_limit, max_nodes
        )
    if direction in {"down", "both"}:
        downstream, down_truncated = _walk(
            conn, repo_id, seeds_down, _DOWNSTREAM_SQL, "down", depth_limit, max_nodes
        )

    files: dict[str, int] = {}
    if target.get("path") and target.get("kind") in INTERNAL_NODE_KINDS:
        files[str(target["path"])] = 0
    for entry in [*upstream, *downstream]:
        node = entry["node"]
        if node["kind"] not in INTERNAL_NODE_KINDS or not node["path"]:
            continue
        path = str(node["path"])
        distance = int(entry["distance"])
        if path not in files or distance < files[path]:
            files[path] = distance
    file_entries = [
        {"path": path, "distance": distance}
        for path, distance in sorted(files.items(), key=lambda item: (item[1], item[0]))
    ]
    entrypoints = [entry for entry in upstream if entry["node"]["kind"] == "Route"]
    conservative_file_expansions = sum(
        1
        for entry in [*upstream, *downstream]
        if entry.get("scope") == "conservative_file"
    )

    return {
        "target_id": node_id,
        "depth": depth_limit,
        "depth_label": "all" if depth_limit is None else str(depth_limit),
        "direction": direction,
        "upstream": upstream,
        "downstream": downstream,
        "files": file_entries,
        "entrypoints": entrypoints,
        "truncated": up_truncated or down_truncated,
        "stats": {
            "upstream": len(upstream),
            "downstream": len(downstream),
            "files": len(file_entries),
            "entrypoints": len(entrypoints),
            "conservative_file_expansions": conservative_file_expansions,
        },
    }


def _call_detail(value: Any) -> str:
    return str(value or "").strip()


def filter_shadowed_external_call_rows(
    rows: list[sqlite3.Row],
) -> list[sqlite3.Row]:
    resolved_details = {
        _call_detail(row["edge_detail"])
        for row in rows
        if row["edge_kind"] == "CALLS"
        and row["kind"] not in {"ExternalSymbol", "ExternalDependency"}
        and _call_detail(row["edge_detail"])
    }
    if not resolved_details:
        return rows
    return [
        row
        for row in rows
        if not (
            row["edge_kind"] == "CALLS"
            and row["kind"] == "ExternalSymbol"
            and _call_detail(row["edge_detail"]) in resolved_details
        )
    ]


def filter_shadowed_external_call_neighbors(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved_details = {
        _call_detail(item["edge"].get("detail"))
        for item in items
        if item.get("edge", {}).get("kind") == "CALLS"
        and (item.get("node") or {}).get("kind")
        not in {"ExternalSymbol", "ExternalDependency"}
        and _call_detail(item["edge"].get("detail"))
    }
    if not resolved_details:
        return items
    return [
        item
        for item in items
        if not (
            item.get("edge", {}).get("kind") == "CALLS"
            and (item.get("node") or {}).get("kind") == "ExternalSymbol"
            and _call_detail(item["edge"].get("detail")) in resolved_details
        )
    ]


def _walk(
    conn: sqlite3.Connection,
    repo_id: str,
    seeds: dict[str, str],
    sql: str,
    direction: str,
    depth: int | None,
    max_nodes: int,
) -> tuple[list[dict[str, Any]], bool]:
    visited: set[str] = set(seeds)
    queue: deque[tuple[str, int, str, str]] = deque(
        (seed_id, 0, "exact", qname) for seed_id, qname in seeds.items()
    )
    entries: list[dict[str, Any]] = []
    while queue:
        current, distance, confidence, current_qname = queue.popleft()
        if depth is not None and distance >= depth:
            continue
        rows = filter_shadowed_external_call_rows(
            conn.execute(sql, (repo_id, current, max_nodes + 1)).fetchall()
        )
        for row in rows:
            other = row["id"]
            if other in visited:
                continue
            visited.add(other)
            combined = weakest_confidence(confidence, row["edge_confidence"])
            if len(entries) >= max_nodes:
                return entries, True
            entries.append(
                {
                    "node": compact_radius_node(row),
                    "distance": distance + 1,
                    "direction": direction,
                    "via": row["edge_kind"],
                    "via_qname": current_qname,
                    "detail": row["edge_detail"] or "",
                    "confidence": combined,
                    "scope": edge_scope(row["edge_kind"]),
                }
            )
            if row["kind"] in EXPANDABLE_NODE_KINDS:
                queue.append((other, distance + 1, combined, row["qname"]))
    return entries, False


def impact_targets(
    conn: sqlite3.Connection, target: str, repo_id: str | None, limit: int
) -> list[dict[str, Any]]:
    node = get_node(conn, target)
    if node and (repo_id is None or node.get("repo_id") == repo_id):
        return [node]
    symbols = find_symbol(conn, target, repo_id=repo_id, limit=limit)
    if symbols:
        return symbols
    return search_nodes(conn, target, repo_id=repo_id, limit=limit)


def impact_report(
    conn: sqlite3.Connection,
    target: dict[str, Any],
    neighbor_limit: int = 200,
    depth: int | None = None,
    max_nodes: int = 1000,
    direction: str = "both",
) -> dict[str, Any]:
    neighbors = get_neighbors(conn, str(target["id"]), limit=neighbor_limit)
    incoming = filter_shadowed_external_call_neighbors(neighbors["incoming"])
    outgoing = filter_shadowed_external_call_neighbors(neighbors["outgoing"])
    callers = [item for item in incoming if item["edge"]["kind"] == "CALLS"]
    callees = [item for item in outgoing if item["edge"]["kind"] == "CALLS"]
    radius = blast_radius(
        conn,
        str(target["id"]),
        depth=depth,
        max_nodes=max_nodes,
        direction=direction,
    )
    files = [entry["path"] for entry in radius.get("files", [])]
    stats = radius.get("stats", {})
    return {
        "target": target,
        "summary": {
            "callers": len(callers),
            "callees": len(callees),
            "incoming": len(incoming),
            "outgoing": len(outgoing),
            "files": len(files),
            "upstream": stats.get("upstream", 0),
            "downstream": stats.get("downstream", 0),
            "entrypoints": stats.get("entrypoints", 0),
            "depth": radius.get("depth", depth),
            "depth_label": radius.get("depth_label", "all"),
            "direction": direction,
            "truncated": radius.get("truncated", False),
        },
        "files": files,
        "callers": callers,
        "callees": callees,
        "blast_radius": radius,
        "incoming": incoming,
        "outgoing": outgoing,
    }
