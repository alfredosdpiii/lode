from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from typing import Any

from lode.config import kuzu_path, sqlite_path
from lode.indexer import index_repo
from lode.kuzu_store import sync_from_sqlite
from lode.storage import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def write_kuzu_fixture(repo: Path) -> None:
    (repo / "app.py").write_text(
        (
            "from service import UserService, audit_user\n\n"
            "def create_user(name):\n"
            "    service = UserService()\n"
            "    audit_user(name)\n"
            "    return service.save_user(name)\n"
        ),
        encoding="utf-8",
    )
    (repo / "service.py").write_text(
        (
            "class UserService:\n"
            "    def save_user(self, name):\n"
            '        return {"name": name}\n\n'
            "def audit_user(name):\n"
            "    return name\n"
        ),
        encoding="utf-8",
    )


def sqlite_counts(data_dir: Path) -> dict[str, int]:
    with closing(connect(sqlite_path(data_dir))) as conn:
        return {
            "repos": conn.execute("SELECT COUNT(*) FROM repos").fetchone()[0],
            "nodes": conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0],
            "edges": conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0],
        }


def require_kuzu() -> Any:
    try:
        return __import__("kuzu")
    except ImportError as exc:
        raise unittest.SkipTest("kuzu optional dependency is not installed") from exc


def open_kuzu(path: Path) -> Any:
    kuzu = require_kuzu()
    return kuzu.Connection(kuzu.Database(str(path)))


def fetch_one(kconn: Any, query: str) -> list[Any]:
    result = kconn.execute(query)
    if not result.has_next():
        raise AssertionError(f"Kuzu query returned no rows: {query}")
    return result.get_next()


class KuzuStoreTests(unittest.TestCase):
    def test_sync_from_sqlite_preserves_counts_schema_and_local_path(self) -> None:
        require_kuzu()
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_kuzu_fixture(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                payload = sync_from_sqlite(conn, kuzu_path(data_dir))

            counts = sqlite_counts(data_dir)
            self.assertEqual(payload["nodes"], counts["nodes"])
            self.assertEqual(payload["edges"], counts["edges"])
            self.assertTrue(Path(payload["kuzu_path"]).resolve().is_relative_to(data_dir.resolve()))

            kconn = open_kuzu(Path(payload["kuzu_path"]))
            self.assertEqual(fetch_one(kconn, "MATCH (n:Node) RETURN count(n)")[0], counts["nodes"])
            self.assertEqual(
                fetch_one(kconn, "MATCH (:Node)-[e:LINK]->(:Node) RETURN count(e)")[0],
                counts["edges"],
            )
            node = fetch_one(
                kconn,
                (
                    "MATCH (n:Node) RETURN n.id, n.repo_id, n.kind, n.name, n.qname, "
                    "n.path, n.start_line, n.end_line, n.signature, n.doc, "
                    "n.confidence, n.extra_json LIMIT 1"
                ),
            )
            self.assertEqual(len(node), 12)
            self.assertTrue(node[0])
            self.assertTrue(node[1])
            self.assertIn(
                node[2],
                {"Class", "ExternalDependency", "ExternalSymbol", "File", "Function", "Method"},
            )
            self.assertTrue(node[5].endswith(".py"))
            self.assertIsInstance(node[6], int)
            self.assertIsInstance(node[7], int)
            self.assertTrue(json.loads(node[11]) is not None)

            rel = fetch_one(
                kconn,
                (
                    "MATCH (:Node)-[e:LINK]->(:Node) "
                    "RETURN e.kind, e.confidence, e.detail, e.repo_id LIMIT 1"
                ),
            )
            self.assertEqual(len(rel), 4)
            self.assertTrue(rel[0])
            self.assertTrue(rel[1])
            self.assertTrue(rel[3])

    def test_index_sync_kuzu_json_contract_uses_sqlite_counts(self) -> None:
        require_kuzu()
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_kuzu_fixture(repo)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "index",
                    str(repo),
                    "--sync-kuzu",
                    "--json",
                ],
                check=True,
                capture_output=True,
                cwd=PROJECT_ROOT,
                text=True,
                timeout=120,
            )

            payload = json.loads(result.stdout)
            self.assertTrue(payload["ok"])
            self.assertIn("kuzu", payload)
            self.assertEqual(payload["kuzu"]["nodes"], payload["nodes"])
            self.assertEqual(payload["kuzu"]["edges"], payload["edges"])
            self.assertTrue(
                Path(payload["kuzu"]["kuzu_path"]).resolve().is_relative_to(data_dir.resolve())
            )

    def test_resync_removes_deleted_file_rows(self) -> None:
        require_kuzu()
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_kuzu_fixture(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                first = sync_from_sqlite(conn, kuzu_path(data_dir))
            self.assertEqual(first["nodes"], sqlite_counts(data_dir)["nodes"])
            self.assertGreater(first["edges"], 0)

            (repo / "app.py").unlink()
            index_repo(repo, sqlite_path(data_dir))
            current_counts = sqlite_counts(data_dir)

            with closing(connect(sqlite_path(data_dir))) as conn:
                second = sync_from_sqlite(conn, kuzu_path(data_dir))
            self.assertEqual(second["nodes"], current_counts["nodes"])
            self.assertEqual(second["edges"], current_counts["edges"])

            kconn = open_kuzu(Path(second["kuzu_path"]))
            self.assertEqual(
                fetch_one(kconn, "MATCH (n:Node) RETURN count(n)")[0], current_counts["nodes"]
            )
            self.assertEqual(
                fetch_one(kconn, "MATCH (:Node)-[e:LINK]->(:Node) RETURN count(e)")[0],
                current_counts["edges"],
            )
            self.assertEqual(
                fetch_one(kconn, "MATCH (n:Node) WHERE n.path = 'app.py' RETURN count(n)")[0],
                0,
            )

            with closing(connect(sqlite_path(data_dir))) as conn:
                third = sync_from_sqlite(conn, kuzu_path(data_dir))
            self.assertEqual(third["nodes"], current_counts["nodes"])
            self.assertEqual(third["edges"], current_counts["edges"])

    def test_cli_missing_kuzu_dependency_is_actionable_and_preserves_sqlite(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_kuzu_fixture(repo)
            index_repo(repo, sqlite_path(data_dir))
            before = sqlite_counts(data_dir)

            env = os.environ.copy()
            env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
            env["PYTHONNOUSERSITE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "-S",
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "kuzu-sync",
                    "--json",
                ],
                capture_output=True,
                cwd=PROJECT_ROOT,
                env=env,
                text=True,
                timeout=120,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            payload = json.loads(result.stderr)
            self.assertFalse(payload["ok"])
            self.assertIn("Kuzu is not installed", payload["error"])
            self.assertIn("uv sync --extra kuzu", payload["error"])
            self.assertEqual(sqlite_counts(data_dir), before)


if __name__ == "__main__":
    unittest.main()
