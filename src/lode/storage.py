from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from .config import repo_id_for_root, sqlite_path
from .model import Edge, FileIndex, Node


class LodeConnection(sqlite3.Connection):
    _lode_cache: dict[tuple[Any, ...], Any]
    _lode_cache_order: list[tuple[Any, ...]]


_QUERY_CACHE_MAX = 128


def connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), factory=LodeConnection)
    conn._lode_cache = {}
    conn._lode_cache_order = []
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS repos (
            id TEXT PRIMARY KEY,
            root TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            indexed_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS files (
            repo_id TEXT NOT NULL,
            path TEXT NOT NULL,
            abspath TEXT NOT NULL,
            language TEXT NOT NULL,
            size INTEGER NOT NULL,
            mtime REAL NOT NULL DEFAULT 0,
            content_hash TEXT NOT NULL,
            generated INTEGER NOT NULL DEFAULT 0,
            indexed_at REAL NOT NULL,
            PRIMARY KEY (repo_id, path),
            FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS nodes (
            id TEXT PRIMARY KEY,
            repo_id TEXT NOT NULL,
            owner_path TEXT NOT NULL,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            qname TEXT NOT NULL,
            path TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            signature TEXT NOT NULL DEFAULT '',
            doc TEXT NOT NULL DEFAULT '',
            confidence TEXT NOT NULL,
            content_hash TEXT NOT NULL DEFAULT '',
            extra_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_nodes_repo_kind ON nodes(repo_id, kind);
        CREATE INDEX IF NOT EXISTS idx_nodes_repo_name ON nodes(repo_id, name);
        CREATE INDEX IF NOT EXISTS idx_nodes_repo_path ON nodes(repo_id, path);

        CREATE TABLE IF NOT EXISTS edges (
            repo_id TEXT NOT NULL,
            owner_path TEXT NOT NULL,
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            kind TEXT NOT NULL,
            confidence TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (repo_id, owner_path, src, dst, kind, detail),
            FOREIGN KEY (repo_id) REFERENCES repos(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(repo_id, src, kind);
        CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(repo_id, dst, kind);

        CREATE TABLE IF NOT EXISTS embedding_queue (
            node_id TEXT PRIMARY KEY,
            repo_id TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            queued_at REAL NOT NULL,
            embedded_at REAL
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            node_id TEXT PRIMARY KEY,
            repo_id TEXT NOT NULL,
            dims INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            model TEXT NOT NULL,
            embedded_at REAL NOT NULL
        );
        """
    )
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS node_fts USING fts5("
        "node_id UNINDEXED, qname, name, signature, doc, path, "
        "tokenize='porter')"
    )
    conn.commit()
    try:
        conn.execute("ALTER TABLE files ADD COLUMN mtime REAL NOT NULL DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    conn.commit()


def upsert_repo(conn: sqlite3.Connection, root: Path) -> str:
    resolved = root.expanduser().resolve()
    repo_id = repo_id_for_root(resolved)
    conn.execute(
        """
        INSERT INTO repos(id, root, name, indexed_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
          root=excluded.root,
          name=excluded.name,
          indexed_at=excluded.indexed_at
        """,
        (repo_id, str(resolved), resolved.name, time.time()),
    )
    return repo_id


def replace_file_index(conn: sqlite3.Connection, repo_id: str, file_index: FileIndex) -> None:
    with conn:
        replace_file_index_rows(conn, repo_id, file_index)


def replace_file_index_rows(conn: sqlite3.Connection, repo_id: str, file_index: FileIndex) -> None:
    replace_file_indexes_rows(conn, repo_id, [file_index])


def replace_file_indexes_rows(
    conn: sqlite3.Connection, repo_id: str, file_indexes: list[FileIndex]
) -> None:
    if not file_indexes:
        return
    now = time.time()
    paths = [file_index.path for file_index in file_indexes]
    old_node_ids: list[str] = []
    for path_group in chunked(paths, 400):
        placeholders = ",".join("?" for _ in path_group)
        old_node_ids.extend(
            row["id"]
            for row in conn.execute(
                f"SELECT id FROM nodes WHERE repo_id = ? AND owner_path IN ({placeholders})",  # nosec B608
                [repo_id, *path_group],
            )
        )
    if old_node_ids:
        conn.executemany(
            "DELETE FROM node_fts WHERE node_id = ?",
            [(node_id,) for node_id in old_node_ids],
        )
    for path_group in chunked(paths, 400):
        placeholders = ",".join("?" for _ in path_group)
        conn.execute(
            f"DELETE FROM edges WHERE repo_id = ? AND owner_path IN ({placeholders})",  # nosec B608
            [repo_id, *path_group],
        )
        conn.execute(
            f"DELETE FROM nodes WHERE repo_id = ? AND owner_path IN ({placeholders})",  # nosec B608
            [repo_id, *path_group],
        )
    file_values = [
        (
            repo_id,
            file_index.path,
            file_index.abspath,
            file_index.language,
            file_index.size,
            file_index.mtime,
            file_index.content_hash,
            1 if file_index.generated else 0,
            now,
        )
        for file_index in file_indexes
    ]
    conn.executemany(
        """
        INSERT INTO files(repo_id, path, abspath, language, size, mtime, content_hash, generated, indexed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(repo_id, path) DO UPDATE SET
          abspath=excluded.abspath,
          language=excluded.language,
          size=excluded.size,
          mtime=excluded.mtime,
          content_hash=excluded.content_hash,
          generated=excluded.generated,
          indexed_at=excluded.indexed_at
        """,
        file_values,
    )
    node_values: list[tuple[Any, ...]] = []
    fts_values: list[tuple[Any, ...]] = []
    queue_values: list[tuple[Any, ...]] = []
    edge_values: list[tuple[Any, ...]] = []
    for file_index in file_indexes:
        deduped_nodes = dedupe_nodes(file_index.nodes)
        for node in deduped_nodes:
            node_values.append(
                (
                    node.id,
                    repo_id,
                    file_index.path,
                    node.kind,
                    node.name,
                    node.qname,
                    node.path,
                    node.start_line,
                    node.end_line,
                    node.signature,
                    node.doc,
                    node.confidence,
                    node.content_hash,
                    json.dumps(node.extra, sort_keys=True),
                )
            )
            fts_values.append((node.id, node.qname, node.name, node.signature, node.doc, node.path))
            if (
                node.kind in {"Function", "Method", "Class", "Route", "DocSection"}
                and node.content_hash
            ):
                queue_values.append((node.id, repo_id, node.content_hash, now))
        deduped_edges = dedupe_edges(file_index.edges)
        edge_values.extend(
            (
                repo_id,
                file_index.path,
                edge.src,
                edge.dst,
                edge.kind,
                edge.confidence,
                edge.detail,
            )
            for edge in deduped_edges
        )
    if node_values:
        conn.executemany(
            """
            INSERT INTO nodes(
                id, repo_id, owner_path, kind, name, qname, path, start_line, end_line,
                signature, doc, confidence, content_hash, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            node_values,
        )
    if fts_values:
        conn.executemany(
            "INSERT INTO node_fts(node_id, qname, name, signature, doc, path) VALUES (?, ?, ?, ?, ?, ?)",
            fts_values,
        )
    if queue_values:
        conn.executemany(
            """
            INSERT INTO embedding_queue(node_id, repo_id, content_hash, queued_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
              content_hash=excluded.content_hash,
              queued_at=CASE
                WHEN embedding_queue.content_hash != excluded.content_hash THEN excluded.queued_at
                ELSE embedding_queue.queued_at
              END,
              embedded_at=CASE
                WHEN embedding_queue.content_hash != excluded.content_hash THEN NULL
                ELSE embedding_queue.embedded_at
              END
            """,
            queue_values,
        )
    if edge_values:
        conn.executemany(
            """
            INSERT OR IGNORE INTO edges(repo_id, owner_path, src, dst, kind, confidence, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            edge_values,
        )


def dedupe_nodes(nodes: list[Node]) -> list[Node]:
    seen: set[str] = set()
    out: list[Node] = []
    for node in nodes:
        if node.id in seen:
            continue
        seen.add(node.id)
        out.append(node)
    return out


def dedupe_edges(edges: list[Edge]) -> list[Edge]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[Edge] = []
    for edge in edges:
        key = (edge.src, edge.dst, edge.kind, edge.detail)
        if key in seen:
            continue
        seen.add(key)
        out.append(edge)
    return out


def chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def remove_missing_files(conn: sqlite3.Connection, repo_id: str, live_paths: set[str]) -> int:
    with conn:
        return remove_missing_file_rows(conn, repo_id, live_paths)


def remove_missing_file_rows(conn: sqlite3.Connection, repo_id: str, live_paths: set[str]) -> int:
    rows = list(conn.execute("SELECT path FROM files WHERE repo_id = ?", (repo_id,)))
    removed = 0
    for row in rows:
        path = row["path"]
        if path in live_paths:
            continue
        old_node_ids = [
            node_row["id"]
            for node_row in conn.execute(
                "SELECT id FROM nodes WHERE repo_id = ? AND owner_path = ?",
                (repo_id, path),
            )
        ]
        if old_node_ids:
            conn.executemany(
                "DELETE FROM node_fts WHERE node_id = ?",
                [(node_id,) for node_id in old_node_ids],
            )
        conn.execute(
            "DELETE FROM edges WHERE repo_id = ? AND owner_path = ?",
            (repo_id, path),
        )
        conn.execute(
            "DELETE FROM nodes WHERE repo_id = ? AND owner_path = ?",
            (repo_id, path),
        )
        conn.execute("DELETE FROM files WHERE repo_id = ? AND path = ?", (repo_id, path))
        removed += 1
    return removed


def list_repos(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT r.id, r.root, r.name, r.indexed_at,
               COUNT(DISTINCT f.path) AS files,
               COUNT(DISTINCT n.id) AS nodes
        FROM repos r
        LEFT JOIN files f ON f.repo_id = r.id
        LEFT JOIN nodes n ON n.repo_id = r.id
        GROUP BY r.id
        ORDER BY r.indexed_at DESC
        """
    )
    return [dict(row) for row in rows]


def repo_filter(conn: sqlite3.Connection, repo_path: str | None) -> str | None:
    if not repo_path:
        return None
    root = Path(repo_path).expanduser().resolve()
    repo_id = repo_id_for_root(root)
    exists = conn.execute("SELECT 1 FROM repos WHERE id = ?", (repo_id,)).fetchone()
    return repo_id if exists else None


def fts_query(query: str) -> str:
    import re

    tokens = re.findall(r"[A-Za-z0-9_./:-]+", query)
    if not tokens:
        return ""
    return " OR ".join(tokens[:12])


def cache_lookup(conn: sqlite3.Connection, namespace: str, parts: tuple[Any, ...]) -> Any | None:
    if not isinstance(conn, LodeConnection):
        return None
    data_version = conn.execute("PRAGMA data_version").fetchone()[0]
    key = (namespace, conn.total_changes, data_version, *parts)
    return conn._lode_cache.get(key)


def cache_store(
    conn: sqlite3.Connection, namespace: str, parts: tuple[Any, ...], value: Any
) -> None:
    if not isinstance(conn, LodeConnection):
        return
    data_version = conn.execute("PRAGMA data_version").fetchone()[0]
    key = (namespace, conn.total_changes, data_version, *parts)
    if key not in conn._lode_cache:
        conn._lode_cache_order.append(key)
    conn._lode_cache[key] = value
    while len(conn._lode_cache_order) > _QUERY_CACHE_MAX:
        old_key = conn._lode_cache_order.pop(0)
        conn._lode_cache.pop(old_key, None)


def copy_node_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in items]


def copy_neighbors(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "node": dict(payload["node"]) if payload.get("node") else None,
        "outgoing": [
            {
                "edge": dict(item["edge"]),
                "node": dict(item["node"]) if item.get("node") else None,
            }
            for item in payload.get("outgoing", [])
        ],
        "incoming": [
            {
                "edge": dict(item["edge"]),
                "node": dict(item["node"]) if item.get("node") else None,
            }
            for item in payload.get("incoming", [])
        ],
    }


def search_nodes(
    conn: sqlite3.Connection,
    query: str,
    repo_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    cache_parts = (query, repo_id, limit)
    cached = cache_lookup(conn, "search_nodes", cache_parts)
    if cached is not None:
        return copy_node_dicts(cached)

    fts = fts_query(query)
    if fts:
        try:
            if repo_id:
                rows = conn.execute(
                    """
                    SELECT n.*, bm25(node_fts) AS rank
                    FROM node_fts
                    JOIN nodes n ON n.id = node_fts.node_id
                    WHERE node_fts MATCH ? AND n.repo_id = ?
                    ORDER BY CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END, rank
                    LIMIT ?
                    """,
                    (fts, repo_id, limit),
                )
            else:
                rows = conn.execute(
                    """
                    SELECT n.*, bm25(node_fts) AS rank
                    FROM node_fts
                    JOIN nodes n ON n.id = node_fts.node_id
                    WHERE node_fts MATCH ?
                    ORDER BY CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END, rank
                    LIMIT ?
                    """,
                    (fts, limit),
                )
            results = [row_to_node_dict(row) for row in rows]
            if results:
                cache_store(conn, "search_nodes", cache_parts, results)
                return copy_node_dicts(results)
        except sqlite3.OperationalError:
            pass

    like = f"%{query}%"
    if repo_id:
        rows = conn.execute(
            """
            SELECT *, 0.0 AS rank FROM nodes
            WHERE repo_id = ? AND (qname LIKE ? OR name LIKE ? OR path LIKE ? OR doc LIKE ?)
            ORDER BY
              CASE WHEN kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
              CASE WHEN name = ? THEN 0 ELSE 1 END,
              length(qname)
            LIMIT ?
            """,
            (repo_id, like, like, like, like, query, limit),
        )
    else:
        rows = conn.execute(
            """
            SELECT *, 0.0 AS rank FROM nodes
            WHERE qname LIKE ? OR name LIKE ? OR path LIKE ? OR doc LIKE ?
            ORDER BY
              CASE WHEN kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
              CASE WHEN name = ? THEN 0 ELSE 1 END,
              length(qname)
            LIMIT ?
            """,
            (like, like, like, like, query, limit),
        )
    results = [row_to_node_dict(row) for row in rows]
    cache_store(conn, "search_nodes", cache_parts, results)
    return copy_node_dicts(results)


def find_symbol(
    conn: sqlite3.Connection,
    name: str,
    repo_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    cache_parts = (name, repo_id, limit)
    cached = cache_lookup(conn, "find_symbol", cache_parts)
    if cached is not None:
        return copy_node_dicts(cached)

    lowered = name.lower()
    if repo_id:
        rows = conn.execute(
            """
            SELECT *, 0.0 AS rank FROM nodes
            WHERE repo_id = ? AND (lower(name) = ? OR lower(qname) = ? OR lower(qname) LIKE ?)
            ORDER BY
              CASE WHEN kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
              CASE WHEN lower(name) = ? THEN 0 ELSE 1 END,
              length(qname)
            LIMIT ?
            """,
            [repo_id, lowered, lowered, f"%{lowered}%", lowered, limit],
        )
    else:
        rows = conn.execute(
            """
            SELECT *, 0.0 AS rank FROM nodes
            WHERE lower(name) = ? OR lower(qname) = ? OR lower(qname) LIKE ?
            ORDER BY
              CASE WHEN kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
              CASE WHEN lower(name) = ? THEN 0 ELSE 1 END,
              length(qname)
            LIMIT ?
            """,
            [lowered, lowered, f"%{lowered}%", lowered, limit],
        )
    results = [row_to_node_dict(row) for row in rows]
    cache_store(conn, "find_symbol", cache_parts, results)
    return copy_node_dicts(results)


def get_node(conn: sqlite3.Connection, node_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT *, 0.0 AS rank FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row_to_node_dict(row) if row else None


def get_neighbors(conn: sqlite3.Connection, node_id: str, limit: int = 80) -> dict[str, Any]:
    cache_parts = (node_id, limit)
    cached = cache_lookup(conn, "get_neighbors", cache_parts)
    if cached is not None:
        return copy_neighbors(cached)

    node = get_node(conn, node_id)
    if not node:
        payload: dict[str, Any] = {"node": None, "outgoing": [], "incoming": []}
        cache_store(conn, "get_neighbors", cache_parts, payload)
        return copy_neighbors(payload)
    repo_id = str(node["repo_id"])
    outgoing_rows = conn.execute(
        """
        SELECT e.kind AS edge_kind, e.confidence AS edge_confidence, e.detail, n.*, 0.0 AS rank
        FROM edges e
        LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
        WHERE e.repo_id = ? AND e.src = ?
        ORDER BY
          CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
          CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
          n.path,
          n.start_line
        LIMIT ?
        """,
        (repo_id, node_id, limit),
    )
    incoming_rows = conn.execute(
        """
        SELECT e.kind AS edge_kind, e.confidence AS edge_confidence, e.detail, n.*, 0.0 AS rank
        FROM edges e
        LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.src
        WHERE e.repo_id = ? AND e.dst = ?
        ORDER BY
          CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
          CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
          n.path,
          n.start_line
        LIMIT ?
        """,
        (repo_id, node_id, limit),
    )
    payload = {
        "node": node,
        "outgoing": [neighbor_row_to_dict(row) for row in outgoing_rows],
        "incoming": [neighbor_row_to_dict(row) for row in incoming_rows],
    }
    cache_store(conn, "get_neighbors", cache_parts, payload)
    return copy_neighbors(payload)


def row_to_node_dict(row: sqlite3.Row) -> dict[str, Any]:
    if row is None:
        return {}
    data = dict(row)
    extra_json = data.pop("extra_json", "{}") or "{}"
    try:
        data["extra"] = json.loads(extra_json)
    except json.JSONDecodeError:
        data["extra"] = {}
    return data


def pending_embedding_nodes(conn: sqlite3.Connection, limit: int = 32) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT n.*, 0.0 AS rank
        FROM embedding_queue q
        JOIN nodes n ON n.id = q.node_id
        WHERE q.embedded_at IS NULL
        ORDER BY q.queued_at
        LIMIT ?
        """,
        (limit,),
    )
    return [row_to_node_dict(row) for row in rows]


def upsert_embedding(
    conn: sqlite3.Connection,
    node_id: str,
    repo_id: str,
    vector: list[float],
    model: str,
    embedded_at: float | None = None,
) -> None:
    now = embedded_at if embedded_at is not None else time.time()
    vector_json = json.dumps(vector)
    with conn:
        conn.execute(
            """
            INSERT INTO embeddings(node_id, repo_id, dims, vector_json, model, embedded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
              repo_id=excluded.repo_id,
              dims=excluded.dims,
              vector_json=excluded.vector_json,
              model=excluded.model,
              embedded_at=excluded.embedded_at
            """,
            (node_id, repo_id, len(vector), vector_json, model, now),
        )
        conn.execute(
            "UPDATE embedding_queue SET embedded_at = ? WHERE node_id = ?",
            (now, node_id),
        )


def embedding_counts(conn: sqlite3.Connection) -> dict[str, int]:
    queued = conn.execute(
        "SELECT COUNT(*) FROM embedding_queue WHERE embedded_at IS NULL"
    ).fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    return {"queued": int(queued), "embedded": int(embedded)}


def neighbor_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = row_to_node_dict(row)
    return {
        "edge": {
            "kind": data.pop("edge_kind"),
            "confidence": data.pop("edge_confidence"),
            "detail": data.pop("detail"),
        },
        "node": data if data.get("id") else None,
    }
