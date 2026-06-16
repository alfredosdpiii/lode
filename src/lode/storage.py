from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

from .config import repo_id_for_root, sqlite_path
from .model import Edge, FileIndex, Node

_QUERY_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]+")
STRUCTURED_FTS_TOKEN_LIMIT = 8


def connect(path: Path | None = None) -> sqlite3.Connection:
    db_path = path or sqlite_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
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
        CREATE INDEX IF NOT EXISTS idx_nodes_lower_name ON nodes(lower(name));
        CREATE INDEX IF NOT EXISTS idx_nodes_lower_qname ON nodes(lower(qname));

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

        CREATE INDEX IF NOT EXISTS idx_embedding_queue_pending
        ON embedding_queue(repo_id, embedded_at, queued_at);

        CREATE INDEX IF NOT EXISTS idx_embeddings_repo
        ON embeddings(repo_id);
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
    deduped_nodes_by_path = {
        file_index.path: dedupe_nodes(file_index.nodes) for file_index in file_indexes
    }
    deduped_edges_by_path = {
        file_index.path: dedupe_edges(file_index.edges) for file_index in file_indexes
    }
    new_node_ids = {node.id for nodes in deduped_nodes_by_path.values() for node in nodes}
    stale_node_ids = [node_id for node_id in old_node_ids if node_id not in new_node_ids]
    delete_embedding_artifacts(conn, stale_node_ids)
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
        deduped_nodes = deduped_nodes_by_path[file_index.path]
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
            if node.kind not in {"ExternalSymbol", "ExternalDependency"}:
                fts_values.append(
                    (node.id, node.qname, node.name, node.signature, node.doc, node.path)
                )
            if (
                node.kind in {"Function", "Method", "Class", "Route", "DocSection"}
                and node.content_hash
            ):
                queue_values.append((node.id, repo_id, node.content_hash, now))
        deduped_edges = deduped_edges_by_path[file_index.path]
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
        execute_values(
            conn,
            """
            INSERT INTO nodes(
                id, repo_id, owner_path, kind, name, qname, path, start_line, end_line,
                signature, doc, confidence, content_hash, extra_json
            )
            """,
            node_values,
        )
    if fts_values:
        execute_values(
            conn,
            "INSERT INTO node_fts(node_id, qname, name, signature, doc, path)",
            fts_values,
        )
    if queue_values:
        execute_values(
            conn,
            """
            INSERT INTO embedding_queue(node_id, repo_id, content_hash, queued_at)
            """,
            queue_values,
            """
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
        )
    if edge_values:
        execute_values(
            conn,
            """
            INSERT OR IGNORE INTO edges(repo_id, owner_path, src, dst, kind, confidence, detail)
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


def chunked_rows(rows: list[tuple[Any, ...]], size: int) -> list[list[tuple[Any, ...]]]:
    return [rows[index : index + size] for index in range(0, len(rows), size)]


def execute_values(
    conn: sqlite3.Connection,
    insert_sql: str,
    rows: list[tuple[Any, ...]],
    suffix_sql: str = "",
) -> None:
    if not rows:
        return
    width = len(rows[0])
    row_placeholder = "(" + ", ".join("?" for _ in range(width)) + ")"
    max_rows = max(1, min(400, 12_000 // max(1, width)))
    for row_group in chunked_rows(rows, max_rows):
        placeholders = ", ".join(row_placeholder for _ in row_group)
        params = [value for row in row_group for value in row]
        conn.execute(
            f"{insert_sql} VALUES {placeholders} {suffix_sql}",  # nosec B608
            params,
        )


def delete_embedding_artifacts(conn: sqlite3.Connection, node_ids: list[str]) -> None:
    if not node_ids:
        return
    for node_group in chunked(node_ids, 400):
        placeholders = ",".join("?" for _ in node_group)
        conn.execute(
            f"DELETE FROM embedding_queue WHERE node_id IN ({placeholders})",  # nosec B608
            node_group,
        )
        conn.execute(
            f"DELETE FROM embeddings WHERE node_id IN ({placeholders})",  # nosec B608
            node_group,
        )


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
            delete_embedding_artifacts(conn, old_node_ids)
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


def fts_query(query: str, operator: str = "OR") -> str:
    tokens = query_tokens(query)
    return fts_query_from_tokens(tokens, operator=operator)


def query_tokens(query: str) -> list[str]:
    return _QUERY_TOKEN_RE.findall(query)


def fts_query_from_tokens(tokens: list[str], operator: str = "OR") -> str:
    if not tokens:
        return ""
    separator = f" {operator} " if operator else " "
    return separator.join(tokens[:12])


def path_fts_query(query: str) -> str:
    return path_fts_query_from_tokens(query_tokens(query))


def path_fts_query_from_tokens(tokens: list[str]) -> str:
    if len(tokens) < 3:
        return ""
    return " OR ".join(f"path: {token}" for token in tokens[:12])


def column_fts_query(query: str) -> str:
    return column_fts_query_from_tokens(query_tokens(query))


def column_fts_query_from_tokens(tokens: list[str]) -> str:
    if not tokens:
        return ""
    columns = ("name", "qname", "path")
    return " OR ".join(f"{column}: {token}" for token in tokens[:12] for column in columns)


def external_like_terms(query: str) -> list[str]:
    return external_like_terms_from_tokens(query_tokens(query))


def external_like_terms_from_tokens(tokens: list[str]) -> list[str]:
    if len(tokens) > STRUCTURED_FTS_TOKEN_LIMIT:
        return []
    return [token.lower() for token in tokens[:12]]


def external_node_like_rows(
    conn: sqlite3.Connection,
    query: str,
    repo_id: str | None,
    limit: int,
    tokens: list[str] | None = None,
) -> Any:
    terms = (
        external_like_terms_from_tokens(tokens)
        if tokens is not None
        else external_like_terms(query)
    )
    if not terms:
        return []
    clauses = []
    params: list[Any] = []
    for term in terms:
        clauses.append("(lower(n.name) LIKE ? OR lower(n.qname) LIKE ? OR lower(n.path) LIKE ?)")
        like = f"%{term}%"
        params.extend([like, like, like])
    filter_sql = " OR ".join(clauses)
    if repo_id:
        return conn.execute(
            f"""
            SELECT n.*, 0.0 AS rank
            FROM nodes n
            WHERE n.repo_id = ?
              AND n.kind IN ('ExternalSymbol', 'ExternalDependency')
              AND ({filter_sql})
            ORDER BY
              CASE WHEN lower(n.name) = ? THEN 0 ELSE 1 END,
              length(n.qname), n.path, n.start_line
            LIMIT ?
            """,  # nosec B608
            [repo_id, *params, query.lower(), limit],
        )
    return conn.execute(
        f"""
        SELECT n.*, 0.0 AS rank
        FROM nodes n
        WHERE n.kind IN ('ExternalSymbol', 'ExternalDependency')
          AND ({filter_sql})
        ORDER BY
          CASE WHEN lower(n.name) = ? THEN 0 ELSE 1 END,
          length(n.qname), n.path, n.start_line
        LIMIT ?
        """,  # nosec B608
        [*params, query.lower(), limit],
    )


def append_unique_nodes(
    rows: Any,
    results: list[dict[str, Any]],
    seen_ids: set[str],
    limit: int,
) -> None:
    for row in rows:
        node = row_to_node_dict(row)
        node_id = str(node["id"])
        if node_id in seen_ids:
            continue
        results.append(node)
        seen_ids.add(node_id)
        if len(results) >= limit:
            break


def search_path_priority(path: str) -> int:
    if path.startswith("src/"):
        return 0
    if path.startswith(("scripts/", "benchmarks/")):
        return 1
    if path.startswith("tests/"):
        return 2
    if path in {"README.md", "AGENTS.md"} or path.startswith(("docs/", "skills/")):
        return 3
    return 1


def search_kind_priority(kind: str) -> int:
    if kind == "File":
        return 0
    if kind == "Class":
        return 1
    if kind in {"Function", "Method"}:
        return 2
    if kind == "Route":
        return 3
    if kind == "DocSection":
        return 4
    return 5


def search_result_sort_key(node: dict[str, Any]) -> tuple[int, float, int, str, int, str]:
    return (
        search_path_priority(str(node.get("path") or "")),
        float(node.get("rank") or 0.0),
        search_kind_priority(str(node.get("kind") or "")),
        str(node.get("path") or ""),
        int(node.get("start_line") or 0),
        str(node.get("qname") or ""),
    )


def unique_nodes_in_order(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for node in candidates:
        node_id = str(node["id"])
        if node_id in seen_ids:
            continue
        results.append(node)
        seen_ids.add(node_id)
        if len(results) >= limit:
            break
    return results


def unique_node_count(candidates: list[dict[str, Any]], limit: int) -> int:
    count = 0
    seen_ids: set[str] = set()
    for node in candidates:
        node_id = str(node["id"])
        if node_id in seen_ids:
            continue
        seen_ids.add(node_id)
        count += 1
        if count >= limit:
            break
    return count


def stable_search_results(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    results = unique_nodes_in_order(candidates, limit)
    if not results:
        return results
    top_priority = search_path_priority(str(results[0].get("path") or ""))
    best_priority = min(search_path_priority(str(node.get("path") or "")) for node in candidates)
    if best_priority < top_priority:
        promoted = min(
            (
                node
                for node in candidates
                if search_path_priority(str(node.get("path") or "")) == best_priority
            ),
            key=search_result_sort_key,
        )
        promoted_id = str(promoted["id"])
        remaining = [node for node in candidates if str(node["id"]) != promoted_id]
        return [promoted, *unique_nodes_in_order(remaining, max(0, limit - 1))]
    return results


def search_nodes(
    conn: sqlite3.Connection,
    query: str,
    repo_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    tokens = query_tokens(query)
    use_structured_fts_probes = len(tokens) <= STRUCTURED_FTS_TOKEN_LIMIT
    strict_fts = fts_query_from_tokens(tokens, operator="") if use_structured_fts_probes else ""
    broad_fts = fts_query_from_tokens(tokens)
    path_fts = path_fts_query_from_tokens(tokens) if use_structured_fts_probes else ""
    column_fts = column_fts_query_from_tokens(tokens) if use_structured_fts_probes else ""
    if strict_fts or broad_fts:
        candidates: list[dict[str, Any]] = []
        if path_fts:
            try:
                if repo_id:
                    path_rows = conn.execute(
                        """
                        SELECT n.*, 0.0 AS rank
                        FROM node_fts
                        JOIN nodes n ON n.id = node_fts.node_id
                        WHERE node_fts MATCH ?
                          AND n.repo_id = ?
                          AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                        LIMIT ?
                        """,
                        (path_fts, repo_id, limit),
                    )
                else:
                    path_rows = conn.execute(
                        """
                        SELECT n.*, 0.0 AS rank
                        FROM node_fts
                        JOIN nodes n ON n.id = node_fts.node_id
                        WHERE node_fts MATCH ?
                          AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                        LIMIT ?
                        """,
                        (path_fts, limit),
                    )
                candidates.extend(row_to_node_dict(row) for row in path_rows)
            except sqlite3.OperationalError:
                pass

        if unique_node_count(candidates, limit) < max(1, limit // 2):
            candidates = []

        column_candidates: list[dict[str, Any]] = []
        if column_fts and not candidates:
            try:
                if repo_id:
                    column_rows = conn.execute(
                        """
                        SELECT n.*, 0.0 AS rank
                        FROM node_fts
                        JOIN nodes n ON n.id = node_fts.node_id
                        WHERE node_fts MATCH ?
                          AND n.repo_id = ?
                          AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                        LIMIT ?
                        """,
                        (column_fts, repo_id, limit),
                    )
                else:
                    column_rows = conn.execute(
                        """
                        SELECT n.*, 0.0 AS rank
                        FROM node_fts
                        JOIN nodes n ON n.id = node_fts.node_id
                        WHERE node_fts MATCH ?
                          AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                        LIMIT ?
                        """,
                        (column_fts, limit),
                    )
                column_candidates = [row_to_node_dict(row) for row in column_rows]
                if unique_node_count(column_candidates, limit) >= limit:
                    candidates = column_candidates
            except sqlite3.OperationalError:
                pass

        if strict_fts and not candidates:
            try:
                if repo_id:
                    strict_rows = conn.execute(
                        """
                        SELECT n.*, bm25(node_fts) AS rank
                        FROM node_fts
                        JOIN nodes n ON n.id = node_fts.node_id
                        WHERE node_fts MATCH ?
                          AND n.repo_id = ?
                          AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (strict_fts, repo_id, limit),
                    )
                else:
                    strict_rows = conn.execute(
                        """
                        SELECT n.*, bm25(node_fts) AS rank
                        FROM node_fts
                        JOIN nodes n ON n.id = node_fts.node_id
                        WHERE node_fts MATCH ?
                          AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                        ORDER BY rank
                        LIMIT ?
                        """,
                        (strict_fts, limit),
                    )
                candidates.extend(row_to_node_dict(row) for row in strict_rows)
            except sqlite3.OperationalError:
                pass

        if broad_fts and broad_fts != strict_fts and unique_node_count(candidates, limit) < limit:
            try:
                current_count = unique_node_count(candidates, limit)
                if current_count < max(1, limit // 2) and column_fts:
                    if column_candidates:
                        candidates.extend(column_candidates)
                    else:
                        fill_limit = limit
                        if repo_id:
                            column_rows = conn.execute(
                                """
                                SELECT n.*, 0.0 AS rank
                                FROM node_fts
                                JOIN nodes n ON n.id = node_fts.node_id
                                WHERE node_fts MATCH ?
                                  AND n.repo_id = ?
                                  AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                                LIMIT ?
                                """,
                                (column_fts, repo_id, fill_limit),
                            )
                        else:
                            column_rows = conn.execute(
                                """
                                SELECT n.*, 0.0 AS rank
                                FROM node_fts
                                JOIN nodes n ON n.id = node_fts.node_id
                                WHERE node_fts MATCH ?
                                  AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                                LIMIT ?
                                """,
                                (column_fts, fill_limit),
                            )
                        candidates.extend(row_to_node_dict(row) for row in column_rows)
                current_count = unique_node_count(candidates, limit)
                if current_count < limit:
                    if current_count >= max(1, limit // 2):
                        fill_limit = max(1, limit - current_count + 2)
                    else:
                        fill_limit = limit
                    if repo_id:
                        broad_rows = conn.execute(
                            """
                            SELECT n.*, 0.0 AS rank
                            FROM node_fts
                            JOIN nodes n ON n.id = node_fts.node_id
                            WHERE node_fts MATCH ?
                              AND n.repo_id = ?
                              AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                            LIMIT ?
                            """,
                            (broad_fts, repo_id, fill_limit),
                        )
                    else:
                        broad_rows = conn.execute(
                            """
                            SELECT n.*, 0.0 AS rank
                            FROM node_fts
                            JOIN nodes n ON n.id = node_fts.node_id
                            WHERE node_fts MATCH ?
                              AND n.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
                            LIMIT ?
                            """,
                            (broad_fts, fill_limit),
                        )
                    candidates.extend(row_to_node_dict(row) for row in broad_rows)
            except sqlite3.OperationalError:
                pass

        results = stable_search_results(candidates, limit)

        seen_ids = {str(node["id"]) for node in results}
        if len(results) < limit and broad_fts:
            fill_limit = limit - len(results)
            external_rows = external_node_like_rows(conn, query, repo_id, fill_limit, tokens)
            append_unique_nodes(external_rows, results, seen_ids, limit)

        if results:
            return results

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
    return results


def find_symbol(
    conn: sqlite3.Connection,
    name: str,
    repo_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    lowered = name.lower()
    if repo_id:
        exact_rows = conn.execute(
            """
            SELECT *, 0.0 AS rank FROM nodes
            WHERE repo_id = ? AND (lower(name) = ? OR lower(qname) = ?)
            ORDER BY
              CASE WHEN kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
              CASE WHEN lower(name) = ? THEN 0 ELSE 1 END,
              length(qname)
            LIMIT ?
            """,
            [repo_id, lowered, lowered, lowered, limit],
        )
    else:
        exact_rows = conn.execute(
            """
            SELECT *, 0.0 AS rank FROM nodes
            WHERE lower(name) = ? OR lower(qname) = ?
            ORDER BY
              CASE WHEN kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
              CASE WHEN lower(name) = ? THEN 0 ELSE 1 END,
              length(qname)
            LIMIT ?
            """,
            [lowered, lowered, lowered, limit],
        )
    results = [row_to_node_dict(row) for row in exact_rows]
    if results:
        return results

    like = f"%{lowered}%"
    if repo_id:
        fallback_rows = conn.execute(
            """
            SELECT *, 0.0 AS rank FROM nodes
            WHERE repo_id = ?
              AND lower(qname) LIKE ?
              AND lower(name) != ?
              AND lower(qname) != ?
            ORDER BY
              CASE WHEN kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
              length(qname)
            LIMIT ?
            """,
            [repo_id, like, lowered, lowered, limit],
        )
    else:
        fallback_rows = conn.execute(
            """
            SELECT *, 0.0 AS rank FROM nodes
            WHERE lower(qname) LIKE ?
              AND lower(name) != ?
              AND lower(qname) != ?
            ORDER BY
              CASE WHEN kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
              length(qname)
            LIMIT ?
            """,
            [like, lowered, lowered, limit],
        )
    results.extend(row_to_node_dict(row) for row in fallback_rows)
    return results


def get_node(conn: sqlite3.Connection, node_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT *, 0.0 AS rank FROM nodes WHERE id = ?", (node_id,)).fetchone()
    return row_to_node_dict(row) if row else None


_NEIGHBOR_CONFIDENCE_ORDER = {"resolved": 0, "strong": 1}
_NEIGHBOR_ORDER_SQL = """
        ORDER BY
          CASE WHEN n.kind IN ('ExternalSymbol', 'ExternalDependency') THEN 1 ELSE 0 END,
          CASE e.confidence WHEN 'resolved' THEN 0 WHEN 'strong' THEN 1 ELSE 2 END,
          COALESCE(n.path, ''),
          CASE
            WHEN n.id IS NULL THEN 0
            WHEN COALESCE(n.start_line, 0) < 1 THEN 1
            ELSE n.start_line
          END
        """
_NEIGHBOR_NODE_COLUMNS_SQL = """
               n.id, n.repo_id, n.owner_path, n.kind, n.name, n.qname, n.path,
               n.start_line, n.end_line, n.signature, n.doc, n.confidence,
               n.content_hash, n.extra_json, 0.0
        """
_OUTGOING_NEIGHBORS_SQL = (
    """
        SELECT e.kind, e.confidence, e.detail,
        """
    + _NEIGHBOR_NODE_COLUMNS_SQL
    + """
        FROM edges e
        LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
        WHERE e.repo_id = ? AND e.src = ?
        """
)
_INCOMING_NEIGHBORS_SQL = (
    """
        SELECT e.kind, e.confidence, e.detail,
        """
    + _NEIGHBOR_NODE_COLUMNS_SQL
    + """
        FROM edges e
        LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.src
        WHERE e.repo_id = ? AND e.dst = ?
        """
)
_NEIGHBOR_NODE_LOOKUP_SQL = """
        SELECT id, repo_id, owner_path, kind, name, qname, path,
               start_line, end_line, signature, doc, confidence, content_hash,
               extra_json, 0.0
        FROM nodes
        WHERE id = ?
        """
_NEIGHBOR_EDGE_KIND = 0
_NEIGHBOR_EDGE_CONFIDENCE = 1
_NEIGHBOR_EDGE_DETAIL = 2
_NEIGHBOR_NODE_ID = 3
_NEIGHBOR_REPO_ID = 4
_NEIGHBOR_OWNER_PATH = 5
_NEIGHBOR_KIND = 6
_NEIGHBOR_NAME = 7
_NEIGHBOR_QNAME = 8
_NEIGHBOR_PATH = 9
_NEIGHBOR_START_LINE = 10
_NEIGHBOR_END_LINE = 11
_NEIGHBOR_SIGNATURE = 12
_NEIGHBOR_DOC = 13
_NEIGHBOR_CONFIDENCE = 14
_NEIGHBOR_CONTENT_HASH = 15
_NEIGHBOR_EXTRA_JSON = 16
_NEIGHBOR_RANK = 17
_NODE_ROW_ID = 0
_NODE_ROW_REPO_ID = 1
_NODE_ROW_OWNER_PATH = 2
_NODE_ROW_KIND = 3
_NODE_ROW_NAME = 4
_NODE_ROW_QNAME = 5
_NODE_ROW_PATH = 6
_NODE_ROW_START_LINE = 7
_NODE_ROW_END_LINE = 8
_NODE_ROW_SIGNATURE = 9
_NODE_ROW_DOC = 10
_NODE_ROW_CONFIDENCE = 11
_NODE_ROW_CONTENT_HASH = 12
_NODE_ROW_EXTRA_JSON = 13
_NODE_ROW_RANK = 14


def neighbor_sort_key(item: dict[str, Any]) -> tuple[int, int, str, int]:
    node = item.get("node") or {}
    edge = item["edge"]
    return (
        1 if node.get("kind") in {"ExternalSymbol", "ExternalDependency"} else 0,
        _NEIGHBOR_CONFIDENCE_ORDER.get(str(edge.get("confidence") or ""), 2),
        str(node.get("path") or ""),
        int(node.get("start_line") or 0),
    )


def neighbor_row_sort_key(row: tuple[Any, ...]) -> tuple[int, int, str, int]:
    node_id = row[_NEIGHBOR_NODE_ID]
    start_line = int(row[_NEIGHBOR_START_LINE] or 0)
    if start_line < 1:
        start_line = 1
    return (
        1 if row[_NEIGHBOR_KIND] in {"ExternalSymbol", "ExternalDependency"} else 0,
        _NEIGHBOR_CONFIDENCE_ORDER.get(str(row[_NEIGHBOR_EDGE_CONFIDENCE] or ""), 2),
        str(row[_NEIGHBOR_PATH] or ""),
        0 if node_id is None else start_line,
    )


def sorted_neighbor_rows(rows: Any, limit: int) -> list[dict[str, Any]]:
    ordered_rows = list(rows)
    ordered_rows.sort(key=neighbor_row_sort_key)
    return [neighbor_row_to_dict(row) for row in ordered_rows[:limit]]


def _bounded_sorted_neighbor_rows(
    conn: sqlite3.Connection,
    query: str,
    repo_id: str,
    node_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return sorted_neighbor_rows(conn.execute(query, (repo_id, node_id)).fetchall(), limit)

    probe_rows = conn.execute(f"{query} LIMIT ?", (repo_id, node_id, limit + 1)).fetchall()
    if len(probe_rows) <= limit:
        return sorted_neighbor_rows(probe_rows, limit)

    rows = conn.execute(
        f"{query}{_NEIGHBOR_ORDER_SQL} LIMIT ?",
        (repo_id, node_id, limit),
    ).fetchall()
    return [neighbor_row_to_dict(row) for row in rows]


def neighbor_node_row_to_dict(row: tuple[Any, ...] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    start_line = int(row[_NODE_ROW_START_LINE] or 0)
    if start_line < 1:
        start_line = 1
    end_line = int(row[_NODE_ROW_END_LINE] or 0)
    if end_line < start_line:
        end_line = start_line
    extra_json = row[_NODE_ROW_EXTRA_JSON] or "{}"
    if extra_json == "{}":
        extra = {}
    else:
        try:
            extra = json.loads(extra_json)
        except json.JSONDecodeError:
            extra = {}
    return {
        "id": row[_NODE_ROW_ID],
        "repo_id": row[_NODE_ROW_REPO_ID],
        "owner_path": row[_NODE_ROW_OWNER_PATH],
        "kind": row[_NODE_ROW_KIND],
        "name": row[_NODE_ROW_NAME],
        "qname": row[_NODE_ROW_QNAME],
        "path": row[_NODE_ROW_PATH],
        "start_line": start_line,
        "end_line": end_line,
        "signature": row[_NODE_ROW_SIGNATURE],
        "doc": row[_NODE_ROW_DOC],
        "confidence": row[_NODE_ROW_CONFIDENCE],
        "content_hash": row[_NODE_ROW_CONTENT_HASH],
        "rank": row[_NODE_ROW_RANK],
        "extra": extra,
    }


def get_neighbors(conn: sqlite3.Connection, node_id: str, limit: int = 80) -> dict[str, Any]:
    row_factory = conn.row_factory
    conn.row_factory = None
    try:
        node = neighbor_node_row_to_dict(
            conn.execute(_NEIGHBOR_NODE_LOOKUP_SQL, (node_id,)).fetchone()
        )
        if not node:
            return {"node": None, "outgoing": [], "incoming": []}
        repo_id = str(node["repo_id"])
        outgoing = _bounded_sorted_neighbor_rows(
            conn,
            _OUTGOING_NEIGHBORS_SQL,
            repo_id,
            node_id,
            limit,
        )
        incoming = _bounded_sorted_neighbor_rows(
            conn,
            _INCOMING_NEIGHBORS_SQL,
            repo_id,
            node_id,
            limit,
        )
    finally:
        conn.row_factory = row_factory
    payload = {"node": node, "outgoing": outgoing, "incoming": incoming}
    return payload


def row_to_node_dict(row: sqlite3.Row) -> dict[str, Any]:
    if row is None:
        return {}
    data = dict(row)
    start_line = int(data.get("start_line") or 0)
    end_line = int(data.get("end_line") or 0)
    data["start_line"] = max(1, start_line)
    data["end_line"] = max(data["start_line"], end_line)
    extra_json = data.pop("extra_json", "{}") or "{}"
    if extra_json == "{}":
        data["extra"] = {}
    else:
        try:
            data["extra"] = json.loads(extra_json)
        except json.JSONDecodeError:
            data["extra"] = {}
    return data


def pending_embedding_nodes(conn: sqlite3.Connection, limit: int = 32) -> list[dict[str, Any]]:
    if limit < 0:
        raise ValueError("Embedding limit must be non-negative")
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
    upsert_embeddings(conn, [(node_id, repo_id, vector, model)], embedded_at=embedded_at)


def upsert_embeddings(
    conn: sqlite3.Connection,
    rows: list[tuple[str, str, list[float], str]],
    embedded_at: float | None = None,
) -> None:
    if not rows:
        return
    now = embedded_at if embedded_at is not None else time.time()
    embedding_values = [
        (node_id, repo_id, len(vector), json.dumps(vector), model, now)
        for node_id, repo_id, vector, model in rows
    ]
    queue_values = [(now, node_id) for node_id, _repo_id, _vector, _model in rows]
    with conn:
        conn.executemany(
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
            embedding_values,
        )
        conn.executemany(
            "UPDATE embedding_queue SET embedded_at = ? WHERE node_id = ?",
            queue_values,
        )


def embedding_counts(conn: sqlite3.Connection) -> dict[str, int]:
    queued = conn.execute(
        "SELECT COUNT(*) FROM embedding_queue WHERE embedded_at IS NULL"
    ).fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    return {"queued": int(queued), "embedded": int(embedded)}


def neighbor_row_to_dict(row: tuple[Any, ...]) -> dict[str, Any]:
    node = None
    if row[_NEIGHBOR_NODE_ID] is not None:
        start_line = int(row[_NEIGHBOR_START_LINE] or 0)
        if start_line < 1:
            start_line = 1
        end_line = int(row[_NEIGHBOR_END_LINE] or 0)
        if end_line < start_line:
            end_line = start_line
        extra_json = row[_NEIGHBOR_EXTRA_JSON] or "{}"
        if extra_json == "{}":
            extra = {}
        else:
            try:
                extra = json.loads(extra_json)
            except json.JSONDecodeError:
                extra = {}
        node = {
            "id": row[_NEIGHBOR_NODE_ID],
            "repo_id": row[_NEIGHBOR_REPO_ID],
            "owner_path": row[_NEIGHBOR_OWNER_PATH],
            "kind": row[_NEIGHBOR_KIND],
            "name": row[_NEIGHBOR_NAME],
            "qname": row[_NEIGHBOR_QNAME],
            "path": row[_NEIGHBOR_PATH],
            "start_line": start_line,
            "end_line": end_line,
            "signature": row[_NEIGHBOR_SIGNATURE],
            "doc": row[_NEIGHBOR_DOC],
            "confidence": row[_NEIGHBOR_CONFIDENCE],
            "content_hash": row[_NEIGHBOR_CONTENT_HASH],
            "rank": row[_NEIGHBOR_RANK],
            "extra": extra,
        }
    return {
        "edge": {
            "kind": row[_NEIGHBOR_EDGE_KIND],
            "confidence": row[_NEIGHBOR_EDGE_CONFIDENCE],
            "detail": row[_NEIGHBOR_EDGE_DETAIL],
        },
        "node": node,
    }
