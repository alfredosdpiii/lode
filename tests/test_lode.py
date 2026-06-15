from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Callable
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from lode.config import sqlite_path
from lode.context import build_context_pack, compact_neighbors
from lode.indexer import index_repo, iter_source_files
from lode.storage import (
    connect,
    embedding_counts,
    find_symbol,
    get_neighbors,
    search_nodes,
)


def trace_statements(conn: sqlite3.Connection, operation: Callable[[], object]) -> list[str]:
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        operation()
    finally:
        conn.set_trace_callback(None)
    return statements


def count_selects(statements: list[str], *fragments: str) -> int:
    count = 0
    for statement in statements:
        normalized = " ".join(statement.split())
        if normalized.upper().startswith("SELECT") and all(
            fragment in normalized for fragment in fragments
        ):
            count += 1
    return count


class LodeIndexTests(unittest.TestCase):
    def test_parse_file_with_explicit_text_matches_disk(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp:
            repo = Path(repo_tmp)
            (repo / "app.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
            from lode.indexer import parse_file

            from_disk = parse_file(repo, repo / "app.py")
            text = (repo / "app.py").read_text(encoding="utf-8")
            from_text = parse_file(repo, repo / "app.py", text=text)
            self.assertEqual(from_disk.content_hash, from_text.content_hash)
            self.assertEqual(len(from_disk.nodes), len(from_text.nodes))
            self.assertEqual(len(from_disk.edges), len(from_text.edges))
            for a, b in zip(from_disk.nodes, from_text.nodes):
                self.assertEqual(a.id, b.id)
                self.assertEqual(a.content_hash, b.content_hash)

    def test_generated_artifact_directories_are_not_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp:
            repo = Path(repo_tmp)
            (repo / "src").mkdir()
            (repo / "src" / "app.py").write_text("def app():\n    return 1\n", encoding="utf-8")
            (repo / "bench-results").mkdir()
            (repo / "bench-results" / "run.json").write_text('{"ok": true}\n', encoding="utf-8")
            (repo / "droid-wiki").mkdir()
            (repo / "droid-wiki" / "overview.md").write_text("# Generated wiki\n", encoding="utf-8")

            indexed = {path.relative_to(repo).as_posix() for path in iter_source_files(repo)}
            self.assertEqual(indexed, {"src/app.py"})

    def test_batch_replace_preserves_nodes_edges_and_queue(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)
            from lode.indexer import index_repo, parse_file
            from lode.storage import connect, replace_file_index, sqlite_path

            # First, do a normal index to get a repo_id
            stats = index_repo(repo, sqlite_path(data_dir))
            repo_id = stats.repo_id

            # Now directly replace a file with batching
            (repo / "app.py").write_text("def new_func():\n    return 42\n", encoding="utf-8")
            file_index = parse_file(repo, repo / "app.py")
            with connect(sqlite_path(data_dir)) as conn:
                replace_file_index(conn, repo_id, file_index)

                # Verify nodes are present with correct IDs and hashes
                node_rows = list(
                    conn.execute(
                        "SELECT id, content_hash FROM nodes WHERE repo_id = ? AND owner_path = ?",
                        (repo_id, "app.py"),
                    )
                )
                self.assertEqual(len(node_rows), len(file_index.nodes))
                for row in node_rows:
                    self.assertTrue(any(row["id"] == n.id for n in file_index.nodes))

                # Verify edges are present with correct details
                edge_rows = list(
                    conn.execute(
                        "SELECT src, dst, kind, detail FROM edges WHERE repo_id = ? AND owner_path = ?",
                        (repo_id, "app.py"),
                    )
                )
                self.assertEqual(len(edge_rows), len(file_index.edges))

                # Verify FTS rows are present
                fts_rows = list(
                    conn.execute(
                        "SELECT node_id FROM node_fts WHERE node_id IN (SELECT id FROM nodes WHERE repo_id = ? AND owner_path = ?)",
                        (repo_id, "app.py"),
                    )
                )
                self.assertEqual(len(fts_rows), len(file_index.nodes))

                # Verify embedding queue has expected rows
                queue_rows = list(
                    conn.execute(
                        "SELECT node_id, content_hash FROM embedding_queue WHERE repo_id = ? AND node_id IN (SELECT id FROM nodes WHERE repo_id = ? AND owner_path = ?)",
                        (repo_id, repo_id, "app.py"),
                    )
                )
                expected_queue = [
                    n
                    for n in file_index.nodes
                    if n.kind in {"Function", "Method", "Class", "Route", "DocSection"}
                    and n.content_hash
                ]
                self.assertEqual(len(queue_rows), len(expected_queue))

    def test_indexes_python_symbols_routes_and_calls(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)

            stats = index_repo(repo, sqlite_path(data_dir))

            self.assertEqual(stats.scanned, 2)
            self.assertGreaterEqual(stats.nodes, 6)
            self.assertGreaterEqual(stats.edges, 5)

            with closing(connect(sqlite_path(data_dir))) as conn:
                functions = find_symbol(conn, "create_user")
                self.assertTrue(functions)
                self.assertEqual(functions[0]["kind"], "Function")

                routes = search_nodes(conn, "POST /users", limit=5)
                self.assertTrue(any(row["kind"] == "Route" for row in routes))

                neighbors = get_neighbors(conn, functions[0]["id"])
                outgoing_edges = {item["edge"]["kind"] for item in neighbors["outgoing"]}
                self.assertIn("CALLS", outgoing_edges)

                context = build_context_pack(conn, "create user route", budget=2000)
                self.assertTrue(context["must_read"])
                self.assertIn("confidence", context)

    def test_find_symbol_prefers_local_definitions_over_external_calls(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            nested = repo / "nested" / "deep"
            nested.mkdir(parents=True)
            (nested / "module.py").write_text(
                "def long_symbol_name():\n    return 1\n",
                encoding="utf-8",
            )
            (repo / "caller.py").write_text(
                "def caller():\n    return long_symbol_name()\n",
                encoding="utf-8",
            )

            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                symbols = find_symbol(conn, "long_symbol_name")
                self.assertTrue(symbols)
                self.assertEqual(symbols[0]["kind"], "Function")
                self.assertEqual(symbols[0]["qname"], "nested.deep.module.long_symbol_name")

    def test_find_symbol_case_insensitive_and_substring_fallback(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "context_tools.py").write_text(
                "def Build_Context_Pack():\n    return 'ok'\n",
                encoding="utf-8",
            )
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                exact = find_symbol(conn, "build_context_pack", limit=5)
                self.assertTrue(exact)
                self.assertEqual(exact[0]["name"], "Build_Context_Pack")
                self.assertEqual(exact[0]["kind"], "Function")

                substring = find_symbol(conn, "context_pack", limit=5)
                self.assertTrue(substring)
                self.assertTrue(
                    any(item["qname"] == "context_tools.Build_Context_Pack" for item in substring)
                )

    def test_context_related_matches_public_neighbor_compaction(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                context = build_context_pack(conn, "create user route", budget=2000, limit=5)
                self.assertTrue(context["related"])
                for related in context["related"]:
                    neighbors = get_neighbors(conn, related["node_id"], limit=16)
                    self.assertEqual(related["incoming"], compact_neighbors(neighbors["incoming"]))
                    self.assertEqual(related["outgoing"], compact_neighbors(neighbors["outgoing"]))

    def test_repeated_query_context_and_graph_lookups_execute_sqlite(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                target = find_symbol(conn, "create_user", limit=5)[0]

                search_nodes(conn, "create user", limit=5)
                search_trace = trace_statements(
                    conn, lambda: search_nodes(conn, "create user", limit=5)
                )
                self.assertGreaterEqual(count_selects(search_trace, "FROM node_fts"), 1)

                symbol_trace = trace_statements(
                    conn, lambda: find_symbol(conn, "create_user", limit=5)
                )
                self.assertGreaterEqual(count_selects(symbol_trace, "FROM nodes"), 1)

                get_neighbors(conn, target["id"], limit=8)
                neighbors_trace = trace_statements(
                    conn, lambda: get_neighbors(conn, target["id"], limit=8)
                )
                self.assertGreaterEqual(count_selects(neighbors_trace, "FROM edges e"), 2)

                build_context_pack(conn, "create user route", budget=2000, limit=5)
                context_trace = trace_statements(
                    conn,
                    lambda: build_context_pack(conn, "create user route", budget=2000, limit=5),
                )
                self.assertGreaterEqual(count_selects(context_trace, "FROM node_fts"), 1)

    def test_neighbors_include_cross_file_callers_after_reindex(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            callee = repo / "callee.py"
            callee.write_text(
                "def target_function():\n    return 'ok'\n",
                encoding="utf-8",
            )
            (repo / "caller.py").write_text(
                "from callee import target_function\n\n"
                "def call_target():\n    return target_function()\n",
                encoding="utf-8",
            )

            stats = index_repo(repo, sqlite_path(data_dir))
            self.assertGreaterEqual(stats.resolved_calls, 1)

            with closing(connect(sqlite_path(data_dir))) as conn:
                target = find_symbol(conn, "target_function")[0]
                old_target_id = target["id"]
                neighbors = get_neighbors(conn, old_target_id)
                self.assert_has_resolved_caller(neighbors, "caller.call_target")

            callee.write_text(
                "\n\ndef target_function():\n    return 'updated'\n",
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))
            self.assertGreaterEqual(stats.resolved_calls, 1)

            with closing(connect(sqlite_path(data_dir))) as conn:
                target = find_symbol(conn, "target_function")[0]
                self.assertNotEqual(target["id"], old_target_id)
                neighbors = get_neighbors(conn, target["id"])
                self.assert_has_resolved_caller(neighbors, "caller.call_target")

    def test_hot_reindex_skips_all_unchanged(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)

            cold = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(cold.scanned, 0)
            self.assertGreater(cold.indexed, 0)
            self.assertGreater(cold.nodes, 0)
            self.assertGreater(cold.edges, 0)
            self.assertEqual(cold.skipped_unchanged, 0)
            self.assertEqual(cold.removed, 0)

            hot = index_repo(repo, sqlite_path(data_dir))
            self.assertEqual(hot.scanned, cold.scanned)
            self.assertEqual(hot.skipped_unchanged, cold.scanned)
            self.assertEqual(hot.indexed, 0)
            self.assertEqual(hot.nodes, 0)
            self.assertEqual(hot.edges, 0)
            self.assertEqual(hot.removed, 0)

    def test_reindex_skips_via_hash_fallback_when_mtime_changes(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)

            cold = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(cold.indexed, 0)

            app_py = repo / "app.py"
            os.utime(app_py, (app_py.stat().st_atime + 1, app_py.stat().st_mtime + 1))

            hot = index_repo(repo, sqlite_path(data_dir))
            self.assertEqual(hot.indexed, 0)
            self.assertEqual(hot.skipped_unchanged, cold.scanned)
            self.assertEqual(hot.nodes, 0)
            self.assertEqual(hot.edges, 0)
            self.assertEqual(hot.removed, 0)

    def test_reindex_same_length_different_content_with_preserved_mtime(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "app.py").write_text(
                "def foo():\n    return 1\n",
                encoding="utf-8",
            )
            cold = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(cold.indexed, 0)
            self.assertGreater(cold.nodes, 0)

            with closing(connect(sqlite_path(data_dir))) as conn:
                self.assertTrue(find_symbol(conn, "foo"))
                old_nodes = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE repo_id = ? AND owner_path = ?",
                    (cold.repo_id, "app.py"),
                ).fetchone()[0]
                self.assertGreater(old_nodes, 0)

            # Rewrite with different content but same length, preserving mtime
            original_mtime = (repo / "app.py").stat().st_mtime
            (repo / "app.py").write_text(
                "def bar():\n    return 2\n",
                encoding="utf-8",
            )
            os.utime(repo / "app.py", (original_mtime, original_mtime))

            hot = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(hot.indexed, 0)
            self.assertGreater(hot.nodes, 0)
            self.assertEqual(hot.removed, 0)

            with closing(connect(sqlite_path(data_dir))) as conn:
                self.assertTrue(find_symbol(conn, "bar"))
                self.assertFalse(find_symbol(conn, "foo"))
                new_nodes = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE repo_id = ? AND owner_path = ?",
                    (cold.repo_id, "app.py"),
                ).fetchone()[0]
                self.assertEqual(new_nodes, old_nodes)

    def test_reindex_unreadable_file_removes_stale_data(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "app.py").write_text(
                "def foo():\n    return 1\n",
                encoding="utf-8",
            )
            cold = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(cold.indexed, 0)

            (repo / "app.py").chmod(0o000)
            try:
                hot = index_repo(repo, sqlite_path(data_dir))
                self.assertEqual(hot.indexed, 0)
                self.assertGreater(hot.removed, 0)

                with closing(connect(sqlite_path(data_dir))) as conn:
                    nodes = conn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE repo_id = ? AND owner_path = ?",
                        (cold.repo_id, "app.py"),
                    ).fetchone()[0]
                    self.assertEqual(nodes, 0)
                    edges = conn.execute(
                        "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND owner_path = ?",
                        (cold.repo_id, "app.py"),
                    ).fetchone()[0]
                    self.assertEqual(edges, 0)
                    files = conn.execute(
                        "SELECT COUNT(*) FROM files WHERE repo_id = ? AND path = ?",
                        (cold.repo_id, "app.py"),
                    ).fetchone()[0]
                    self.assertEqual(files, 0)
            finally:
                (repo / "app.py").chmod(0o644)

    def test_reindex_after_file_change_updates_and_no_stale_nodes(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)

            cold = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(cold.indexed, 0)

            (repo / "app.py").write_text(
                "def new_function():\n    return 42\n",
                encoding="utf-8",
            )

            hot = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(hot.indexed, 0)
            self.assertEqual(hot.removed, 0)

            with closing(connect(sqlite_path(data_dir))) as conn:
                new_fn = find_symbol(conn, "new_function")
                self.assertTrue(new_fn)
                old_fn = find_symbol(conn, "create_user")
                self.assertFalse(old_fn)
                app_nodes = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE repo_id = ? AND owner_path = ?",
                    (cold.repo_id, "app.py"),
                ).fetchone()[0]
                self.assertEqual(app_nodes, 2)
                app_edges = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND owner_path = ?",
                    (cold.repo_id, "app.py"),
                ).fetchone()[0]
                self.assertGreaterEqual(app_edges, 0)

    def test_reindex_after_file_delete_removes_stale_data(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)

            cold = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(cold.indexed, 0)

            (repo / "app.py").unlink()

            hot = index_repo(repo, sqlite_path(data_dir))
            self.assertEqual(hot.indexed, 0)
            self.assertGreater(hot.removed, 0)

            with closing(connect(sqlite_path(data_dir))) as conn:
                nodes = conn.execute(
                    "SELECT COUNT(*) FROM nodes WHERE repo_id = ? AND owner_path = ?",
                    (cold.repo_id, "app.py"),
                ).fetchone()[0]
                self.assertEqual(nodes, 0)
                edges = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND owner_path = ?",
                    (cold.repo_id, "app.py"),
                ).fetchone()[0]
                self.assertEqual(edges, 0)
                files = conn.execute(
                    "SELECT COUNT(*) FROM files WHERE repo_id = ? AND path = ?",
                    (cold.repo_id, "app.py"),
                ).fetchone()[0]
                self.assertEqual(files, 0)
                orphaned_edges = conn.execute(
                    """
                    SELECT COUNT(*) FROM edges e
                    LEFT JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst
                    WHERE e.repo_id = ? AND n.id IS NULL
                    """,
                    (cold.repo_id,),
                ).fetchone()[0]
                self.assertEqual(orphaned_edges, 0)

    def test_reindex_coverage_does_not_shrink(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)

            cold = index_repo(repo, sqlite_path(data_dir))
            self.assertGreater(cold.indexed, 0)

            hot = index_repo(repo, sqlite_path(data_dir))
            self.assertEqual(hot.indexed, 0)

            with closing(connect(sqlite_path(data_dir))) as conn:
                counts = conn.execute(
                    """
                    SELECT COUNT(DISTINCT f.path) AS files, COUNT(DISTINCT n.id) AS nodes,
                           COUNT(DISTINCT e.rowid) AS edges
                    FROM repos r
                    LEFT JOIN files f ON f.repo_id = r.id
                    LEFT JOIN nodes n ON n.repo_id = r.id
                    LEFT JOIN edges e ON e.repo_id = r.id
                    WHERE r.id = ?
                    """,
                    (cold.repo_id,),
                ).fetchone()
                self.assertGreaterEqual(counts["files"], 2)
                self.assertGreaterEqual(counts["nodes"], 6)
                self.assertGreaterEqual(counts["edges"], 5)

    def assert_has_resolved_caller(self, neighbors: dict[str, Any], caller_qname: str) -> None:
        incoming = neighbors["incoming"]
        self.assertTrue(
            any(
                item["edge"]["kind"] == "CALLS"
                and item["edge"]["confidence"] == "resolved"
                and item["node"]["qname"] == caller_qname
                for item in incoming
            ),
            incoming,
        )

    def test_cli_json_output(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)

            index_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "index",
                    str(repo),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            index_payload = json.loads(index_result.stdout)
            self.assertTrue(index_payload["ok"])

            search_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "search",
                    "UserService",
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            search_payload = json.loads(search_result.stdout)
            self.assertTrue(search_payload["ok"])
            self.assertTrue(search_payload["results"])

            impact_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "impact",
                    "audit_user",
                    "--repo",
                    str(repo),
                    "--json",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            impact_payload = json.loads(impact_result.stdout)
            self.assertTrue(impact_payload["ok"])
            self.assertEqual(
                impact_payload["results"][0]["callers"][0]["node"]["qname"],
                "app.create_user",
            )

    def test_cli_embed_persists_vectors_from_local_endpoint(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEmbeddingHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                embed_result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "lode",
                        "--data-dir",
                        str(data_dir),
                        "embed",
                        "--url",
                        url,
                        "--model",
                        "fake-embedding-model",
                        "--limit",
                        "3",
                        "--json",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            embed_payload = json.loads(embed_result.stdout)
            self.assertTrue(embed_payload["ok"])
            self.assertEqual(embed_payload["embedded"], 3)
            with closing(connect(sqlite_path(data_dir))) as conn:
                counts = embedding_counts(conn)
                self.assertEqual(counts["embedded"], 3)
                dims = conn.execute("SELECT MIN(dims), MAX(dims) FROM embeddings").fetchone()
                self.assertEqual(tuple(dims), (3, 3))


class FakeEmbeddingHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        size = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(size).decode("utf-8"))
        inputs = body.get("inputs", [])
        vectors = [
            [float(index), float(index + 1), float(len(text))] for index, text in enumerate(inputs)
        ]
        payload = json.dumps(vectors).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        _ = (format, args)
        return


def write_sample_repo(repo: Path) -> None:
    (repo / "app.py").write_text(
        '''
from fastapi import FastAPI

app = FastAPI()


class UserService:
    """Creates and stores users."""
    def save_user(self, name):
        return {"name": name}


def audit_user(name):
    return name


@app.post("/users")
def create_user(name: str):
    service = UserService()
    audit_user(name)
    return service.save_user(name)
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text(
        "# Sample\n\nThe create user route calls the user service.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()
