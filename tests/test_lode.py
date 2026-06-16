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

from lode.cli import EMBEDDING_TEXT_MAX_INPUT_CHARS, embedding_text
from lode.config import sqlite_path
from lode.context import build_context_pack, compact_neighbors
from lode.indexer import index_repo, iter_source_files
from lode.storage import (
    connect,
    embedding_counts,
    external_like_terms_from_tokens,
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


def run_lode_json(data_dir: Path, *args: str, env: dict[str, str] | None = None) -> dict[str, Any]:
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    result = subprocess.run(
        [sys.executable, "-m", "lode", "--data-dir", str(data_dir), *args],
        check=True,
        capture_output=True,
        env=command_env,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def start_fake_embedding_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    FakeEmbeddingHandler.reset()
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeEmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_port}"


def stop_fake_embedding_server(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class LodeIndexTests(unittest.TestCase):
    def test_embedding_text_caps_large_payloads_while_preserving_identity(self) -> None:
        text = embedding_text(
            {
                "qname": "docs.long_section",
                "signature": "### Long Section",
                "doc": "x" * (EMBEDDING_TEXT_MAX_INPUT_CHARS * 4),
            }
        )

        self.assertLessEqual(len(text), EMBEDDING_TEXT_MAX_INPUT_CHARS)
        self.assertTrue(text.startswith("docs.long_section\n### Long Section\n"))

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
                expected_fts_ids = {
                    n.id
                    for n in file_index.nodes
                    if n.kind not in {"ExternalSymbol", "ExternalDependency"}
                }
                self.assertEqual(len(fts_rows), len(expected_fts_ids))

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

    def test_python_parser_preserves_nested_call_attribution(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "app.py").write_text(
                "\n".join(
                    [
                        "def helper():",
                        "    return 1",
                        "",
                        "def outer():",
                        "    def inner():",
                        "        return helper()",
                        "    return inner()",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                outer = find_symbol(conn, "outer")[0]
                helper = find_symbol(conn, "helper")[0]
                edge = conn.execute(
                    """
                    SELECT confidence, detail FROM edges
                    WHERE src = ? AND dst = ? AND kind = 'CALLS'
                    """,
                    (outer["id"], helper["id"]),
                ).fetchone()
                self.assertIsNotNone(edge)
                self.assertEqual(edge["confidence"], "strong")
                self.assertEqual(edge["detail"], "helper")

    def test_function_local_imports_still_resolve_calls(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "callee.py").write_text(
                "def target():\n    return 'ok'\n",
                encoding="utf-8",
            )
            (repo / "caller.py").write_text(
                "\n".join(
                    [
                        "def caller():",
                        "    from callee import target",
                        "    return target()",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stats = index_repo(repo, sqlite_path(data_dir))
            self.assertGreaterEqual(stats.resolved_calls, 1)

            with closing(connect(sqlite_path(data_dir))) as conn:
                target = find_symbol(conn, "target")[0]
                neighbors = get_neighbors(conn, target["id"])
                self.assert_has_resolved_caller(neighbors, "caller.caller")

    def test_repeated_external_calls_are_persisted_once_per_owner(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "app.py").write_text(
                "\n".join(
                    [
                        "def noisy():",
                        "    print('one')",
                        "    print('two')",
                        "    print('three')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                external_print_nodes = conn.execute(
                    """
                    SELECT COUNT(*) FROM nodes
                    WHERE repo_id = ? AND owner_path = ? AND kind = 'ExternalSymbol' AND name = ?
                    """,
                    (stats.repo_id, "app.py", "print"),
                ).fetchone()[0]
                print_edges = conn.execute(
                    """
                    SELECT COUNT(*) FROM edges
                    WHERE repo_id = ? AND owner_path = ? AND kind = 'CALLS' AND detail = ?
                    """,
                    (stats.repo_id, "app.py", "print"),
                ).fetchone()[0]
                self.assertEqual(external_print_nodes, 1)
                self.assertEqual(print_edges, 1)
                external_fts_rows = conn.execute(
                    """
                    SELECT COUNT(*) FROM node_fts f
                    JOIN nodes n ON n.id = f.node_id
                    WHERE n.repo_id = ? AND n.kind = 'ExternalSymbol' AND n.name = ?
                    """,
                    (stats.repo_id, "print"),
                ).fetchone()[0]
                self.assertEqual(external_fts_rows, 0)
                external_results = search_nodes(conn, "print", limit=5)
                self.assertTrue(
                    any(
                        row["kind"] == "ExternalSymbol" and row["name"] == "print"
                        for row in external_results
                    )
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

    def test_operational_search_and_context_preserve_source_ranking(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            source_dir = repo / "src" / "lode"
            source_dir.mkdir(parents=True)
            (source_dir / "embeddings.py").write_text(
                "def embeddings_url():\n"
                "    return 'http://127.0.0.1:7980'\n\n"
                "def embeddings_model():\n"
                "    return 'Snowflake/snowflake-arctic-embed-s'\n",
                encoding="utf-8",
            )
            (source_dir / "context.py").write_text(
                "def build_context_pack():\n"
                '    """Build context pack for agents."""\n'
                "    return []\n",
                encoding="utf-8",
            )
            (repo / "README.md").write_text(
                "# Architecture\n\n"
                "The operational docs mention the embedding queue explicitly.\n\n"
                "# Context\n\n"
                "The docs also describe how to build context pack outputs.\n",
                encoding="utf-8",
            )
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                embedding_hits = search_nodes(conn, "embedding queue", limit=20)
                self.assertTrue(embedding_hits)
                self.assertEqual(embedding_hits[0]["path"], "src/lode/embeddings.py")

                context = build_context_pack(conn, "build context pack", budget=4000, limit=10)
                self.assertEqual(context["confidence"], "strong")
                self.assertTrue(
                    any(hit["path"] == "src/lode/context.py" for hit in context["top_hits"])
                )

    def test_long_search_query_uses_single_broad_fts_lookup(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            long_query = (
                "create user service save account profile controller repository "
                "validate serialize response payload"
            )
            self.assertEqual(external_like_terms_from_tokens(long_query.split()), [])
            self.assertEqual(external_like_terms_from_tokens(["print"]), ["print"])
            with closing(connect(sqlite_path(data_dir))) as conn:
                results = search_nodes(conn, long_query, limit=5)
                self.assertTrue(results)
                self.assertTrue(any(row["path"] == "app.py" for row in results))

                search_trace = trace_statements(
                    conn, lambda: search_nodes(conn, long_query, limit=5)
                )

        self.assertEqual(count_selects(search_trace, "FROM node_fts"), 1)

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

            server, thread, url = start_fake_embedding_server()
            try:
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
                stop_fake_embedding_server(server, thread)

            embed_payload = json.loads(embed_result.stdout)
            self.assertTrue(embed_payload["ok"])
            self.assertEqual(embed_payload["embedded"], 3)
            with closing(connect(sqlite_path(data_dir))) as conn:
                counts = embedding_counts(conn)
                self.assertEqual(counts["embedded"], 3)
                dims = conn.execute("SELECT MIN(dims), MAX(dims) FROM embeddings").fetchone()
                self.assertEqual(tuple(dims), (3, 3))

    def test_cli_embed_honors_limit_and_no_pending_is_idempotent(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            server, thread, url = start_fake_embedding_server()
            try:
                with closing(connect(sqlite_path(data_dir))) as conn:
                    before = embedding_counts(conn)
                self.assertGreater(before["queued"], 2)

                first = run_lode_json(
                    data_dir,
                    "embed",
                    "--url",
                    url,
                    "--model",
                    "fake-embedding-model",
                    "--limit",
                    "2",
                    "--json",
                )
                self.assertEqual(first["embedded"], 2)
                self.assertEqual(len(FakeEmbeddingHandler.requests), 1)
                self.assertEqual(len(FakeEmbeddingHandler.requests[0]), 2)
                with closing(connect(sqlite_path(data_dir))) as conn:
                    after_first = embedding_counts(conn)
                self.assertEqual(after_first["queued"], before["queued"] - 2)
                self.assertEqual(after_first["embedded"], before["embedded"] + 2)

                remaining = run_lode_json(
                    data_dir,
                    "embed",
                    "--url",
                    url,
                    "--model",
                    "fake-embedding-model",
                    "--limit",
                    "100",
                    "--json",
                )
                self.assertEqual(remaining["embedded"], after_first["queued"])
                self.assertEqual(len(FakeEmbeddingHandler.requests), 2)

                no_pending = run_lode_json(
                    data_dir,
                    "embed",
                    "--url",
                    url,
                    "--model",
                    "fake-embedding-model",
                    "--json",
                )
                self.assertTrue(no_pending["ok"])
                self.assertEqual(no_pending["embedded"], 0)
                self.assertEqual(no_pending["queued"], 0)
                self.assertEqual(no_pending["total_embeddings"], before["queued"])
                self.assertEqual(len(FakeEmbeddingHandler.requests), 2)
            finally:
                stop_fake_embedding_server(server, thread)

    def test_cli_embed_fails_closed_without_local_endpoint(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                before = embedding_counts(conn)
            command_env = os.environ.copy()
            command_env.pop("LODE_EMBEDDINGS_URL", None)
            command_env.pop("KG_EMBEDDINGS_URL", None)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "embed",
                    "--json",
                ],
                capture_output=True,
                env=command_env,
                text=True,
                timeout=120,
            )
            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(result.stderr)
            self.assertFalse(payload["ok"])
            self.assertIn("No local embeddings endpoint configured", payload["error"])
            self.assertNotIn("Traceback", result.stdout + result.stderr)
            with closing(connect(sqlite_path(data_dir))) as conn:
                self.assertEqual(embedding_counts(conn), before)

    def test_cli_embed_rejects_hosted_endpoint_without_network_call(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "embed",
                    "--url",
                    "https://example.com",
                    "--json",
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertNotEqual(result.returncode, 0)
            payload = json.loads(result.stderr)
            self.assertFalse(payload["ok"])
            self.assertIn("must be local", payload["error"])

    def test_changed_content_refreshes_embedding_without_duplicate_rows(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "app.py").write_text(
                "def target():\n    return 1\n",
                encoding="utf-8",
            )
            index_repo(repo, sqlite_path(data_dir))

            server, thread, url = start_fake_embedding_server()
            try:
                first = run_lode_json(
                    data_dir,
                    "embed",
                    "--url",
                    url,
                    "--model",
                    "first-model",
                    "--limit",
                    "1",
                    "--json",
                )
                self.assertEqual(first["embedded"], 1)

                with closing(connect(sqlite_path(data_dir))) as conn:
                    node = find_symbol(conn, "target")[0]
                    node_id = node["id"]
                    before = conn.execute(
                        "SELECT dims, vector_json, model, embedded_at FROM embeddings WHERE node_id = ?",
                        (node_id,),
                    ).fetchone()
                    self.assertIsNotNone(before)

                (repo / "app.py").write_text(
                    "def target():\n    return 2\n",
                    encoding="utf-8",
                )
                index_repo(repo, sqlite_path(data_dir))

                with closing(connect(sqlite_path(data_dir))) as conn:
                    node = find_symbol(conn, "target")[0]
                    self.assertEqual(node["id"], node_id)
                    queue = conn.execute(
                        "SELECT embedded_at FROM embedding_queue WHERE node_id = ?",
                        (node_id,),
                    ).fetchone()
                    self.assertIsNotNone(queue)
                    self.assertIsNone(queue["embedded_at"])
                    row_count = conn.execute(
                        "SELECT COUNT(*) FROM embeddings WHERE node_id = ?",
                        (node_id,),
                    ).fetchone()[0]
                    self.assertEqual(row_count, 1)

                FakeEmbeddingHandler.vector_base = 100.0
                second = run_lode_json(
                    data_dir,
                    "embed",
                    "--url",
                    url,
                    "--model",
                    "second-model",
                    "--limit",
                    "1",
                    "--json",
                )
                self.assertEqual(second["embedded"], 1)
                self.assertEqual(len(FakeEmbeddingHandler.requests), 2)

                with closing(connect(sqlite_path(data_dir))) as conn:
                    rows = list(
                        conn.execute(
                            "SELECT dims, vector_json, model, embedded_at FROM embeddings WHERE node_id = ?",
                            (node_id,),
                        )
                    )
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["dims"], 3)
                self.assertEqual(rows[0]["model"], "second-model")
                self.assertNotEqual(rows[0]["vector_json"], before["vector_json"])
                self.assertGreaterEqual(rows[0]["embedded_at"], before["embedded_at"])
            finally:
                stop_fake_embedding_server(server, thread)

    def test_deleted_nodes_do_not_leave_embedding_artifacts(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "app.py").write_text(
                "def deleted_target():\n    return 1\n",
                encoding="utf-8",
            )
            (repo / "keep.py").write_text(
                "def kept_target():\n    return 2\n",
                encoding="utf-8",
            )
            index_repo(repo, sqlite_path(data_dir))

            server, thread, url = start_fake_embedding_server()
            try:
                payload = run_lode_json(
                    data_dir,
                    "embed",
                    "--url",
                    url,
                    "--model",
                    "fake-embedding-model",
                    "--limit",
                    "10",
                    "--json",
                )
                self.assertGreaterEqual(payload["embedded"], 2)
            finally:
                stop_fake_embedding_server(server, thread)

            with closing(connect(sqlite_path(data_dir))) as conn:
                deleted_node_ids = [
                    row["id"]
                    for row in conn.execute(
                        "SELECT id FROM nodes WHERE owner_path = ?",
                        ("app.py",),
                    )
                ]
                self.assertTrue(deleted_node_ids)

            (repo / "app.py").unlink()
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                orphaned_queue = conn.execute(
                    """
                    SELECT COUNT(*) FROM embedding_queue q
                    LEFT JOIN nodes n ON n.id = q.node_id
                    WHERE n.id IS NULL
                    """
                ).fetchone()[0]
                orphaned_embeddings = conn.execute(
                    """
                    SELECT COUNT(*) FROM embeddings e
                    LEFT JOIN nodes n ON n.id = e.node_id
                    WHERE n.id IS NULL
                    """
                ).fetchone()[0]
                self.assertEqual(orphaned_queue, 0)
                self.assertEqual(orphaned_embeddings, 0)
                for node_id in deleted_node_ids:
                    queued = conn.execute(
                        "SELECT COUNT(*) FROM embedding_queue WHERE node_id = ?",
                        (node_id,),
                    ).fetchone()[0]
                    embedded = conn.execute(
                        "SELECT COUNT(*) FROM embeddings WHERE node_id = ?",
                        (node_id,),
                    ).fetchone()[0]
                    self.assertEqual(queued, 0)
                    self.assertEqual(embedded, 0)


class FakeEmbeddingHandler(BaseHTTPRequestHandler):
    requests: list[list[str]] = []
    vector_base: float = 0.0

    @classmethod
    def reset(cls) -> None:
        cls.requests = []
        cls.vector_base = 0.0

    def do_POST(self) -> None:
        size = int(self.headers.get("content-length") or 0)
        body = json.loads(self.rfile.read(size).decode("utf-8"))
        inputs = body.get("inputs", [])
        type(self).requests.append(list(inputs))
        vectors = [
            [
                type(self).vector_base + float(index),
                type(self).vector_base + float(index + 1),
                float(len(text)),
            ]
            for index, text in enumerate(inputs)
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
