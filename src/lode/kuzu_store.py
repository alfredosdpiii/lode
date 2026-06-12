from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .config import kuzu_path


def sync_from_sqlite(conn: sqlite3.Connection, path: Path | None = None) -> dict[str, Any]:
    try:
        import kuzu  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Kuzu is not installed. Install with `uv sync --extra kuzu` or use Docker."
        ) from exc

    db_path = path or kuzu_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    database = kuzu.Database(str(db_path))
    kconn = kuzu.Connection(database)
    reset_schema(kconn)

    nodes = [dict(row) for row in conn.execute("SELECT * FROM nodes")]
    edges = [dict(row) for row in conn.execute("SELECT * FROM edges")]
    for node in nodes:
        extra_json = node.get("extra_json") or "{}"
        try:
            extra = json.dumps(json.loads(extra_json), sort_keys=True)
        except json.JSONDecodeError:
            extra = "{}"
        kconn.execute(
            "CREATE (:Node {id: $id, repo_id: $repo_id, kind: $kind, name: $name, qname: $qname, path: $path, start_line: $start_line, end_line: $end_line, signature: $signature, doc: $doc, confidence: $confidence, extra_json: $extra_json})",
            {
                "id": node["id"],
                "repo_id": node["repo_id"],
                "kind": node["kind"],
                "name": node["name"],
                "qname": node["qname"],
                "path": node["path"],
                "start_line": node["start_line"],
                "end_line": node["end_line"],
                "signature": node.get("signature") or "",
                "doc": node.get("doc") or "",
                "confidence": node.get("confidence") or "unknown",
                "extra_json": extra,
            },
        )
    inserted_edges = 0
    for edge in edges:
        try:
            kconn.execute(
                "MATCH (s:Node {id: $src}), (d:Node {id: $dst}) CREATE (s)-[:LINK {kind: $kind, confidence: $confidence, detail: $detail, repo_id: $repo_id}]->(d)",
                {
                    "src": edge["src"],
                    "dst": edge["dst"],
                    "kind": edge["kind"],
                    "confidence": edge["confidence"],
                    "detail": edge.get("detail") or "",
                    "repo_id": edge["repo_id"],
                },
            )
            inserted_edges += 1
        except RuntimeError:
            continue
    return {"kuzu_path": str(db_path), "nodes": len(nodes), "edges": inserted_edges}


def reset_schema(kconn: Any) -> None:
    for statement in [
        "DROP TABLE IF EXISTS LINK",
        "DROP TABLE IF EXISTS Node",
        "CREATE NODE TABLE Node(id STRING, repo_id STRING, kind STRING, name STRING, qname STRING, path STRING, start_line INT64, end_line INT64, signature STRING, doc STRING, confidence STRING, extra_json STRING, PRIMARY KEY(id))",
        "CREATE REL TABLE LINK(FROM Node TO Node, kind STRING, confidence STRING, detail STRING, repo_id STRING)",
    ]:
        kconn.execute(statement)
