from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import closing
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from lode.config import sqlite_path
from lode.context import build_context_pack
from lode.indexer import index_repo
from lode.storage import (
    connect,
    embedding_counts,
    find_symbol,
    get_neighbors,
    search_nodes,
)


class LodeIndexTests(unittest.TestCase):
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
