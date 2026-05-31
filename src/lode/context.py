from __future__ import annotations

import sqlite3
from typing import Any

from .storage import get_neighbors, repo_filter, search_nodes


def build_context_pack(
    conn: sqlite3.Connection,
    query: str,
    repo_path: str | None = None,
    budget: int = 6000,
    limit: int = 10,
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
    for hit in hits[:5]:
        neighbors = get_neighbors(conn, hit["id"], limit=16)
        related.append(
            {
                "node_id": hit["id"],
                "qname": hit["qname"],
                "incoming": compact_neighbors(neighbors["incoming"]),
                "outgoing": compact_neighbors(neighbors["outgoing"]),
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
        return "exact"
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

