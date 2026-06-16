from __future__ import annotations

import json
import sqlite3
import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from lode.config import sqlite_path
from lode.storage import connect, get_neighbors, neighbor_sort_key, upsert_repo

HIGH_DEGREE_NEIGHBORS = 650


def normalize_sql(statement: str) -> str:
    return " ".join(statement.split())


def insert_neighbor_graph(
    conn: sqlite3.Connection,
    repo: Path,
    neighbor_count: int = HIGH_DEGREE_NEIGHBORS,
) -> str:
    repo_id = upsert_repo(conn, repo)
    center_id = f"{repo_id}:center:{neighbor_count}"
    node_rows: list[tuple[Any, ...]] = [
        (
            center_id,
            repo_id,
            "center.py",
            "Function",
            "center",
            "center.center",
            "center.py",
            1,
            2,
            "def center(): ...",
            "",
            "strong",
            "center-hash",
            "{}",
        )
    ]
    edge_rows: list[tuple[str, str, str, str, str, str, str]] = []
    confidences = ["heuristic", "strong", "resolved"]
    for index in range(neighbor_count):
        target_id = f"{repo_id}:target:{index}"
        path = f"pkg/{neighbor_count - index:04d}_{index:04d}.py"
        kind = "ExternalSymbol" if index % 17 == 0 else "Function"
        confidence = confidences[index % len(confidences)]
        node_rows.append(
            (
                target_id,
                repo_id,
                path,
                kind,
                f"target_{index}",
                f"pkg.target_{index}",
                path,
                index + 1,
                index + 1,
                f"def target_{index}(): ...",
                "",
                "strong",
                f"target-hash-{index}",
                json.dumps({"index": index}, sort_keys=True),
            )
        )
        edge_rows.append(
            (
                repo_id,
                "center.py",
                center_id,
                target_id,
                "CALLS",
                confidence,
                f"outgoing-{index}",
            )
        )
        edge_rows.append(
            (
                repo_id,
                path,
                target_id,
                center_id,
                "CALLS",
                confidence,
                f"incoming-{index}",
            )
        )
    with conn:
        conn.executemany(
            """
            INSERT INTO nodes(
              id, repo_id, owner_path, kind, name, qname, path, start_line, end_line,
              signature, doc, confidence, content_hash, extra_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            node_rows,
        )
        conn.executemany(
            """
            INSERT INTO edges(repo_id, owner_path, src, dst, kind, confidence, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            edge_rows,
        )
    return center_id


class StorageNeighborTests(unittest.TestCase):
    def test_low_degree_neighbors_skip_count_queries_when_limit_covers_rows(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            with closing(connect(sqlite_path(Path(data_tmp)))) as conn:
                node_id = insert_neighbor_graph(conn, repo, neighbor_count=3)
                statements: list[str] = []
                conn.set_trace_callback(statements.append)
                try:
                    neighbors = get_neighbors(conn, node_id, limit=80)
                finally:
                    conn.set_trace_callback(None)

        self.assertEqual(len(neighbors["outgoing"]), 3)
        self.assertEqual(len(neighbors["incoming"]), 3)
        count_selects = [
            normalize_sql(statement)
            for statement in statements
            if "COUNT(*) FROM edges" in normalize_sql(statement)
        ]
        self.assertEqual(count_selects, [])
        for direction in ("outgoing", "incoming"):
            keys = [neighbor_sort_key(item) for item in neighbors[direction]]
            self.assertEqual(keys, sorted(keys))

    def test_high_degree_neighbors_preserve_public_ordering_when_limited(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            with closing(connect(sqlite_path(Path(data_tmp)))) as conn:
                node_id = insert_neighbor_graph(conn, repo)

                full_neighbors = get_neighbors(conn, node_id, limit=HIGH_DEGREE_NEIGHBORS + 1)
                limited_neighbors = get_neighbors(conn, node_id, limit=13)

        self.assertEqual(full_neighbors["node"]["id"], node_id)
        self.assertEqual(len(limited_neighbors["outgoing"]), 13)
        self.assertEqual(len(limited_neighbors["incoming"]), 13)
        self.assertEqual(
            limited_neighbors["outgoing"],
            full_neighbors["outgoing"][:13],
        )
        self.assertEqual(
            limited_neighbors["incoming"],
            full_neighbors["incoming"][:13],
        )
        for direction in ("outgoing", "incoming"):
            keys = [neighbor_sort_key(item) for item in limited_neighbors[direction]]
            self.assertEqual(keys, sorted(keys))
        for neighbor in limited_neighbors["outgoing"] + limited_neighbors["incoming"]:
            self.assertEqual(set(neighbor.keys()), {"edge", "node"})
            self.assertTrue({"kind", "confidence", "detail"}.issubset(neighbor["edge"]))
            self.assertIsNotNone(neighbor["node"])

    def test_high_degree_neighbors_execute_bounded_edge_queries_for_small_limits(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            with closing(connect(sqlite_path(Path(data_tmp)))) as conn:
                node_id = insert_neighbor_graph(conn, repo)
                statements: list[str] = []
                conn.set_trace_callback(statements.append)
                try:
                    neighbors = get_neighbors(conn, node_id, limit=7)
                finally:
                    conn.set_trace_callback(None)

        self.assertEqual(len(neighbors["outgoing"]), 7)
        self.assertEqual(len(neighbors["incoming"]), 7)
        edge_selects = [
            normalize_sql(statement)
            for statement in statements
            if "FROM edges e" in normalize_sql(statement)
        ]
        self.assertTrue(edge_selects)
        self.assertTrue(
            all(" LIMIT " in statement.upper() for statement in edge_selects),
            edge_selects,
        )
