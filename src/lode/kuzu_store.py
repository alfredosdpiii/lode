from __future__ import annotations

import json
import sqlite3
import tempfile
from csv import writer
from pathlib import Path
from typing import Any

from .config import kuzu_path

NODE_COLUMNS = [
    "id",
    "repo_id",
    "kind",
    "name",
    "qname",
    "path",
    "start_line",
    "end_line",
    "signature",
    "doc",
    "confidence",
    "extra_json",
]
EDGE_COLUMNS = ["src", "dst", "kind", "confidence", "detail", "repo_id"]


def sync_from_sqlite(conn: sqlite3.Connection, path: Path | None = None) -> dict[str, Any]:
    try:
        import kuzu  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Kuzu is not installed. Install with `uv sync --extra kuzu` or "
            "`pip install 'lode-kg[kuzu]'`."
        ) from exc

    db_path = path or kuzu_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    database = kuzu.Database(str(db_path))
    kconn = kuzu.Connection(database)
    reset_schema(kconn)

    with tempfile.TemporaryDirectory(prefix=".lode-kuzu-", dir=db_path.parent) as tmp:
        tmp_path = Path(tmp)
        nodes_csv = tmp_path / "nodes.csv"
        edges_csv = tmp_path / "edges.csv"
        node_count = write_nodes_csv(conn, nodes_csv)
        edge_count = write_edges_csv(conn, edges_csv)
        if node_count:
            kconn.execute(
                f"COPY Node FROM {kuzu_path_literal(nodes_csv)} (HEADER=true, PARALLEL=false)"  # nosec B608
            )
        if edge_count:
            kconn.execute(
                f"COPY LINK FROM {kuzu_path_literal(edges_csv)} (HEADER=true, PARALLEL=false)"  # nosec B608
            )

    projected_nodes = kuzu_count(kconn, "MATCH (n:Node) RETURN count(n)")
    projected_edges = kuzu_count(kconn, "MATCH (:Node)-[e:LINK]->(:Node) RETURN count(e)")
    if projected_nodes != node_count or projected_edges != edge_count:
        raise RuntimeError(
            "Kuzu projection count mismatch: "
            f"SQLite nodes={node_count}, edges={edge_count}; "
            f"Kuzu nodes={projected_nodes}, edges={projected_edges}."
        )
    return {"kuzu_path": str(db_path), "nodes": node_count, "edges": edge_count}


def write_nodes_csv(conn: sqlite3.Connection, path: Path) -> int:
    count = 0
    columns = ", ".join(NODE_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = writer(handle)
        csv_writer.writerow(NODE_COLUMNS)
        for node in conn.execute(f"SELECT {columns} FROM nodes ORDER BY id"):  # nosec B608
            csv_writer.writerow(
                [
                    node["id"],
                    node["repo_id"],
                    node["kind"],
                    node["name"],
                    node["qname"],
                    node["path"],
                    node["start_line"],
                    node["end_line"],
                    node["signature"] or "",
                    node["doc"] or "",
                    node["confidence"] or "unknown",
                    normalized_extra_json(node["extra_json"]),
                ]
            )
            count += 1
    return count


def write_edges_csv(conn: sqlite3.Connection, path: Path) -> int:
    count = 0
    columns = ", ".join(EDGE_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv_writer = writer(handle)
        csv_writer.writerow(EDGE_COLUMNS)
        for edge in conn.execute(
            f"SELECT {columns} FROM edges ORDER BY repo_id, owner_path, src, dst, kind, detail"  # nosec B608
        ):
            csv_writer.writerow(
                [
                    edge["src"],
                    edge["dst"],
                    edge["kind"],
                    edge["confidence"],
                    edge["detail"] or "",
                    edge["repo_id"],
                ]
            )
            count += 1
    return count


def normalized_extra_json(raw: str | None) -> str:
    try:
        return json.dumps(json.loads(raw or "{}"), sort_keys=True)
    except json.JSONDecodeError:
        return "{}"


def kuzu_path_literal(path: Path) -> str:
    escaped = path.as_posix().replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def kuzu_count(kconn: Any, query: str) -> int:
    result = kconn.execute(query)
    if not result.has_next():
        return 0
    return int(result.get_next()[0])


def reset_schema(kconn: Any) -> None:
    for statement in [
        "DROP TABLE IF EXISTS LINK",
        "DROP TABLE IF EXISTS Node",
        "CREATE NODE TABLE Node(id STRING, repo_id STRING, kind STRING, name STRING, qname STRING, path STRING, start_line INT64, end_line INT64, signature STRING, doc STRING, confidence STRING, extra_json STRING, PRIMARY KEY(id))",
        "CREATE REL TABLE LINK(FROM Node TO Node, kind STRING, confidence STRING, detail STRING, repo_id STRING)",
    ]:
        kconn.execute(statement)
