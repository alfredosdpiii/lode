from __future__ import annotations

import sqlite3
from typing import Any

from .storage import repo_filter, search_nodes


def build_context_pack(
    conn: sqlite3.Connection,
    query: str,
    repo_path: str | None = None,
    budget: int = 6000,
    limit: int = 10,
    include_related: bool = True,
) -> dict[str, Any]:
    repo_id = repo_filter(conn, repo_path)
    hits = sorted(search_nodes(conn, query, repo_id=repo_id, limit=limit), key=context_rank)
    must_read: list[dict[str, Any]] = []
    seen_paths: set[tuple[str, int, int]] = set()
    remaining = max(1000, budget)

    for hit in hits:
        if remaining <= 0:
            break
        item = context_item(hit, reason_for_hit(hit, query))
        key = (item["path"], item["start_line"], item["end_line"])
        if key in seen_paths:
            continue
        seen_paths.add(key)
        cost = estimate_item_cost(item)
        if cost > remaining and must_read:
            continue
        must_read.append(item)
        remaining -= cost

    related = []
    if include_related:
        related_neighbors = compact_neighbors_for_hits(conn, hits[:5], limit=16)
        for hit in hits[:5]:
            neighbors = related_neighbors.get(hit["id"], {"incoming": [], "outgoing": []})
            related.append(
                {
                    "node_id": hit["id"],
                    "qname": hit["qname"],
                    "incoming": neighbors["incoming"],
                    "outgoing": neighbors["outgoing"],
                }
            )

    return {
        "query": query,
        "budget": budget,
        "summary": summarize_hits(hits),
        "must_read": must_read,
        "top_hits": [compact_node(hit) for hit in hits],
        "related": related,
        "confidence": aggregate_confidence(hits),
        "notes": [
            "Lode ranks exact/FTS matches first, then expands graph neighbors.",
            "Embeddings are optional and not required for this context pack yet.",
        ],
    }


def context_item(node: dict[str, Any], why: str) -> dict[str, Any]:
    return {
        "node_id": node["id"],
        "kind": node["kind"],
        "name": node["name"],
        "qname": node["qname"],
        "path": node["path"],
        "start_line": node["start_line"],
        "end_line": node["end_line"],
        "signature": node.get("signature") or "",
        "doc": truncate(node.get("doc") or "", 800),
        "why": why,
        "confidence": node.get("confidence") or "unknown",
    }


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": node["id"],
        "kind": node["kind"],
        "qname": node["qname"],
        "path": node["path"],
        "lines": [node["start_line"], node["end_line"]],
        "confidence": node.get("confidence") or "unknown",
    }


def compact_neighbors(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for item in items:
        node = item.get("node") or {}
        if not node:
            continue
        out.append(
            {
                "edge": item["edge"]["kind"],
                "qname": node.get("qname"),
                "kind": node.get("kind"),
                "path": node.get("path"),
                "confidence": item["edge"].get("confidence"),
            }
        )
    return out


def compact_neighbors_for_hits(
    conn: sqlite3.Connection, hits: list[dict[str, Any]], limit: int
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    related: dict[str, dict[str, list[dict[str, Any]]]] = {
        str(hit["id"]): {"incoming": [], "outgoing": []}
        for hit in hits
        if hit.get("id") and hit.get("repo_id")
    }
    if not related:
        return related
    ids_by_repo: dict[str, list[str]] = {}
    for hit in hits:
        node_id = str(hit.get("id") or "")
        repo_id = str(hit.get("repo_id") or "")
        if node_id and repo_id and node_id in related:
            ids_by_repo.setdefault(repo_id, []).append(node_id)
    for repo_id, node_ids in ids_by_repo.items():
        unique_ids = list(dict.fromkeys(node_ids))
        load_compact_neighbor_rows(conn, related, repo_id, unique_ids, "outgoing", limit)
        load_compact_neighbor_rows(conn, related, repo_id, unique_ids, "incoming", limit)
    return related


def load_compact_neighbor_rows(
    conn: sqlite3.Connection,
    related: dict[str, dict[str, list[dict[str, Any]]]],
    repo_id: str,
    node_ids: list[str],
    direction: str,
    limit: int,
) -> None:
    if not node_ids:
        return
    placeholders = ",".join("?" for _ in node_ids)
    if direction == "outgoing":
        sql = f"""
        SELECT e.src AS center_id,
               e.kind AS edge_kind,
               e.confidence AS edge_confidence,
               n.kind AS node_kind,
               n.qname AS node_qname,
               n.path AS node_path,
               n.start_line AS node_start_line
        FROM edges e
        LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
        WHERE e.repo_id = ? AND e.src IN ({placeholders})
        ORDER BY
          e.src,
          CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
          CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
          n.path,
          n.start_line
        """
    else:
        sql = f"""
        SELECT e.dst AS center_id,
               e.kind AS edge_kind,
               e.confidence AS edge_confidence,
               n.kind AS node_kind,
               n.qname AS node_qname,
               n.path AS node_path,
               n.start_line AS node_start_line
        FROM edges e
        LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.src
        WHERE e.repo_id = ? AND e.dst IN ({placeholders})
        ORDER BY
          e.dst,
          CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
          CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
          n.path,
          n.start_line
        """
    rows = conn.execute(
        sql,
        [repo_id, *node_ids],
    )
    counts = {node_id: 0 for node_id in node_ids}
    for row in rows:
        center_id = str(row["center_id"])
        if counts.get(center_id, 0) >= limit or row["node_qname"] is None:
            continue
        related[center_id][direction].append(
            {
                "edge": row["edge_kind"],
                "qname": row["node_qname"],
                "kind": row["node_kind"],
                "path": row["node_path"],
                "confidence": row["edge_confidence"],
            }
        )
        counts[center_id] = counts.get(center_id, 0) + 1


def reason_for_hit(node: dict[str, Any], query: str) -> str:
    query_lower = query.lower()
    if query_lower in (node.get("name") or "").lower():
        return "name match"
    if query_lower in (node.get("qname") or "").lower():
        return "qualified-name match"
    if query_lower in (node.get("path") or "").lower():
        return "path match"
    return "full-text match"


def summarize_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "No indexed nodes matched the query."
    kinds: dict[str, int] = {}
    paths: set[str] = set()
    for hit in hits:
        kinds[hit["kind"]] = kinds.get(hit["kind"], 0) + 1
        paths.add(hit["path"])
    kind_summary = ", ".join(f"{kind}:{count}" for kind, count in sorted(kinds.items()))
    return f"Found {len(hits)} relevant nodes across {len(paths)} files ({kind_summary})."


def aggregate_confidence(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "none"
    confidences = {hit.get("confidence") for hit in hits}
    if confidences <= {"exact"}:
        return "exact" if len(hits) == 1 else "strong"
    if confidences <= {"exact", "strong"}:
        return "strong"
    return "mixed"


def estimate_item_cost(item: dict[str, Any]) -> int:
    return max(120, len(str(item)) // 4)


def truncate(value: str, length: int) -> str:
    if len(value) <= length:
        return value
    return value[: length - 3] + "..."


def context_rank(node: dict[str, Any]) -> tuple[int, float, str]:
    kind_penalty = 1 if node.get("kind") in {"ExternalSymbol", "ExternalDependency"} else 0
    return (kind_penalty, float(node.get("rank") or 0.0), node.get("qname") or "")
