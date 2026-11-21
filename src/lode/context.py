from __future__ import annotations

import math
import re
import sqlite3
from pathlib import PurePosixPath
from typing import Any

from .storage import (
    ENGLISH_STOPWORDS,
    PYTHON_KEYWORD_STOPWORDS,
    repo_filter,
    row_to_node_dict,
    search_nodes,
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+")
_CONTEXT_STOPWORDS = PYTHON_KEYWORD_STOPWORDS | ENGLISH_STOPWORDS | {"py", "ts", "js"}


def build_context_pack(
    conn: sqlite3.Connection,
    query: str,
    repo_path: str | None = None,
    budget: int = 6000,
    limit: int = 10,
    include_related: bool = True,
    from_path: str | None = None,
    exclude_paths: set[str] | None = None,
) -> dict[str, Any]:
    repo_id = repo_filter(conn, repo_path)
    excluded = set(exclude_paths) if exclude_paths else set()
    fetch_limit = max(limit * 3, limit + len(excluded) + 10)
    raw_hits = search_nodes(conn, query, repo_id=repo_id, limit=fetch_limit)

    graph_boosts, graph_nodes = _graph_context_signals(
        conn, repo_id, from_path, raw_hits, excluded, limit
    )
    hits = _merge_and_rank_hits(
        query, raw_hits, graph_boosts, graph_nodes, from_path, excluded, limit
    )

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
            "Lode ranks weighted BM25 matches with identifier specificity and graph proximity.",
            "Embeddings are optional and not required for this context pack yet.",
        ],
    }


def _graph_context_signals(
    conn: sqlite3.Connection,
    repo_id: str | None,
    from_path: str | None,
    raw_hits: list[dict[str, Any]],
    excluded: set[str],
    limit: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    if not from_path:
        return {}, []
    boosts: dict[str, float] = {}
    extra_nodes: list[dict[str, Any]] = []
    seen_ids: set[str] = {str(hit["id"]) for hit in raw_hits}

    edge_rows = _load_from_path_targets(conn, repo_id, from_path, max(limit * 2, 24))
    for edge_kind, confidence, node in edge_rows:
        path = str(node.get("path") or "")
        if not path or path in excluded:
            continue
        weight = _edge_boost_weight(edge_kind, confidence)
        node_id = str(node["id"])
        boosts[node_id] = max(boosts.get(node_id, 0.0), weight)
        boosts[f"path:{path}"] = max(boosts.get(f"path:{path}", 0.0), weight * 0.75)
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            extra_nodes.append(node)
    return boosts, extra_nodes


def _load_from_path_targets(
    conn: sqlite3.Connection,
    repo_id: str | None,
    from_path: str,
    limit: int,
) -> list[tuple[str, str, dict[str, Any]]]:
    repo_clause = "AND e.repo_id = ?" if repo_id else ""
    sql = f"""
        SELECT e.kind AS edge_kind, e.confidence AS edge_confidence, n.*
        FROM edges e
        JOIN nodes src ON src.repo_id = e.repo_id AND src.id = e.src
        JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
        WHERE src.path = ?
          {repo_clause}
          AND n.path != ?
          AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency', 'ParseError')
          AND e.kind IN ('CALLS', 'IMPORTS', 'EXTENDS')
        ORDER BY
          CASE e.kind WHEN 'CALLS' THEN 0 WHEN 'EXTENDS' THEN 1 ELSE 2 END,
          CASE e.confidence WHEN 'exact' THEN 0 WHEN 'strong' THEN 1 WHEN 'resolved' THEN 2 ELSE 3 END,
          n.path,
          n.start_line
        LIMIT ?
    """  # nosec B608
    params: tuple[Any, ...] = (
        (from_path, repo_id, from_path, limit) if repo_id else (from_path, from_path, limit)
    )
    out: list[tuple[str, str, dict[str, Any]]] = []
    for row in conn.execute(sql, params):
        node = row_to_node_dict(row)
        node["rank"] = 0.0
        out.append((str(row["edge_kind"]), str(row["edge_confidence"]), node))
    return out


def _edge_boost_weight(edge_kind: str, confidence: str) -> float:
    kind_base = {"CALLS": 4.0, "EXTENDS": 4.0, "IMPORTS": 3.5}.get(edge_kind, 2.0)
    conf_scale = {"exact": 1.0, "strong": 1.0, "resolved": 0.9, "heuristic": 0.6}.get(
        confidence, 0.5
    )
    return kind_base * conf_scale


def _merge_and_rank_hits(
    query: str,
    raw_hits: list[dict[str, Any]],
    graph_boosts: dict[str, float],
    graph_nodes: list[dict[str, Any]],
    from_path: str | None,
    excluded: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    query_weights, exact_terms = _query_term_weights(query)
    from_parts = _path_dir_parts(from_path) if from_path else ()
    merged = _dedupe_candidate_nodes((*raw_hits, *graph_nodes), excluded)
    if not merged:
        return []

    node_fields = [_extract_node_fields(node) for node in merged]
    term_idf = _candidate_pool_idf(node_fields)
    scored = [
        (
            _composite_rank(
                node, fields, query_weights, exact_terms, term_idf, graph_boosts, from_parts
            ),
            node,
        )
        for node, fields in zip(merged, node_fields, strict=True)
    ]
    scored.sort(key=lambda pair: pair[0])
    return _diversify_by_path([node for _, node in scored], limit)


def _dedupe_candidate_nodes(
    nodes: tuple[dict[str, Any], ...], excluded: set[str]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in nodes:
        node_id = str(node.get("id") or "")
        path = str(node.get("path") or "")
        if not node_id or node_id in seen_ids or path in excluded:
            continue
        seen_ids.add(node_id)
        merged.append(node)
    return merged


def _diversify_by_path(nodes: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    primary: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    per_path: dict[str, int] = {}
    for node in nodes:
        path = str(node.get("path") or "")
        count = per_path.get(path, 0)
        if count < 2:
            primary.append(node)
            per_path[path] = count + 1
        else:
            overflow.append(node)
    return (primary + overflow)[:limit]


def _extract_node_fields(node: dict[str, Any]) -> tuple[set[str], set[str], set[str], set[str]]:
    name_raw = str(node.get("name") or "")
    path_raw = str(node.get("path") or "")
    sig_raw = str(node.get("signature") or "")
    exact = {tok.lower() for tok in _TOKEN_RE.findall(f"{name_raw} {sig_raw}") if len(tok) > 1}
    return (
        _field_token_set(name_raw),
        _field_token_set(path_raw),
        _field_token_set(sig_raw),
        exact,
    )


def _candidate_pool_idf(
    node_fields: list[tuple[set[str], set[str], set[str], set[str]]],
) -> dict[str, float]:
    total = len(node_fields)
    if total <= 1:
        return {}
    doc_freq: dict[str, int] = {}
    for name_toks, path_toks, sig_toks, _ in node_fields:
        for tok in name_toks | path_toks | sig_toks:
            doc_freq[tok] = doc_freq.get(tok, 0) + 1
    return {tok: math.log((total + 1.0) / (df + 0.5)) + 0.5 for tok, df in doc_freq.items()}


def _composite_rank(
    node: dict[str, Any],
    fields: tuple[set[str], set[str], set[str], set[str]],
    query_weights: dict[str, float],
    exact_terms: dict[str, float],
    term_idf: dict[str, float],
    graph_boosts: dict[str, float],
    from_parts: tuple[str, ...],
) -> tuple[int, float, str]:
    kind_penalty = 1 if node.get("kind") in {"ExternalSymbol", "ExternalDependency"} else 0
    base_rank = float(node.get("rank") or 0.0)
    node_id = str(node.get("id") or "")
    path = str(node.get("path") or "")
    boost = graph_boosts.get(node_id, 0.0) + graph_boosts.get(f"path:{path}", 0.0)
    boost += _lexical_specificity_boost(fields, query_weights, exact_terms, term_idf)
    if from_parts and path:
        boost += _path_proximity_boost(from_parts, _path_dir_parts(path))
    return (kind_penalty, base_rank - boost, str(node.get("qname") or ""))


def _query_term_weights(query: str) -> tuple[dict[str, float], dict[str, float]]:
    weights: dict[str, float] = {}
    exact_weights: dict[str, float] = {}
    pos = 0
    for raw in _TOKEN_RE.findall(query):
        low_raw = raw.lower()
        if low_raw not in _CONTEXT_STOPWORDS and len(low_raw) > 1:
            exact_weights[low_raw] = max(exact_weights.get(low_raw, 0.0), 1.0 / (1.0 + 0.22 * pos))
        for token in _expand_token(raw):
            if token in _CONTEXT_STOPWORDS or len(token) <= 1:
                continue
            w = 1.0 / (1.0 + 0.25 * pos)
            if w > weights.get(token, 0.0):
                weights[token] = w
            pos += 1
            if pos >= 48:
                return weights, exact_weights
    return weights, exact_weights


def _expand_token(raw: str) -> list[str]:
    low_raw = raw.lower()
    out = [low_raw]
    for chunk in raw.split("_"):
        if not chunk:
            continue
        out.extend(p.lower() for p in _CAMEL_RE.findall(chunk) if p.lower() != low_raw)
    return out


def _field_token_set(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _TOKEN_RE.findall(value):
        tokens.update(_expand_token(raw))
    return tokens


def _lexical_specificity_boost(
    fields: tuple[set[str], set[str], set[str], set[str]],
    query_weights: dict[str, float],
    exact_terms: dict[str, float],
    term_idf: dict[str, float],
) -> float:
    if not query_weights:
        return 0.0
    name_tokens, path_tokens, sig_tokens, exact_tokens = fields
    boost = 0.0
    for term, weight in exact_terms.items():
        if term in exact_tokens:
            boost += 5.0 * weight * term_idf.get(term, 1.5)
    for term, weight in query_weights.items():
        idf = term_idf.get(term, 1.5)
        if term in name_tokens:
            boost += 3.5 * weight * idf
        if term in path_tokens:
            boost += 3.0 * weight * idf
        if term in sig_tokens:
            boost += 1.5 * weight * idf
    return boost


def _path_dir_parts(path: str) -> tuple[str, ...]:
    return PurePosixPath(path).parent.parts


def _path_proximity_boost(from_parts: tuple[str, ...], cand_parts: tuple[str, ...]) -> float:
    if not from_parts or not cand_parts:
        return 0.0
    shared = 0
    for a, b in zip(from_parts, cand_parts, strict=False):
        if a != b:
            break
        shared += 1
    if shared == 0:
        return 0.0
    return 4.0 if from_parts == cand_parts else min(3.0, 1.0 * shared)


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
               n.path AS node_path
        FROM edges e
        JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
        WHERE e.repo_id = ? AND e.src IN ({placeholders})
        ORDER BY
          e.src,
          CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
          CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
          n.path,
          n.start_line
        """  # nosec B608
    else:
        sql = f"""
        SELECT e.dst AS center_id,
               e.kind AS edge_kind,
               e.confidence AS edge_confidence,
               n.kind AS node_kind,
               n.qname AS node_qname,
               n.path AS node_path
        FROM edges e
        JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.src
        WHERE e.repo_id = ? AND e.dst IN ({placeholders})
        ORDER BY
          e.dst,
          CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
          CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
          n.path,
          n.start_line
        """  # nosec B608
    row_factory = conn.row_factory
    conn.row_factory = None
    try:
        rows = conn.execute(
            sql,
            [repo_id, *node_ids],
        )
        counts = {node_id: 0 for node_id in node_ids}
        for row in rows:
            center_id = str(row[0])
            if counts.get(center_id, 0) >= limit or row[4] is None:
                continue
            related[center_id][direction].append(
                {
                    "edge": row[1],
                    "qname": row[4],
                    "kind": row[3],
                    "path": row[5],
                    "confidence": row[2],
                }
            )
            counts[center_id] = counts.get(center_id, 0) + 1
    finally:
        conn.row_factory = row_factory


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
