from __future__ import annotations

import unittest
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from lode.config import sqlite_path
from lode.graph import blast_radius, impact_report, impact_targets, resolve_graph
from lode.indexer import index_repo
from lode.storage import connect, find_symbol


def _write_multi_file_repo(repo: Path) -> None:
    models = repo / "models"
    models.mkdir()
    (models / "__init__.py").write_text("", encoding="utf-8")
    (models / "user.py").write_text(
        "class User:\n    def __init__(self, name):\n        self.name = name\n"
        "    def greet(self):\n        return f'hello {self.name}'\n",
        encoding="utf-8",
    )
    services = repo / "services"
    services.mkdir()
    (services / "__init__.py").write_text("", encoding="utf-8")
    (services / "auth.py").write_text(
        "from models.user import User\n\n"
        "def authenticate(name):\n"
        "    user = User(name)\n"
        "    return user.greet()\n",
        encoding="utf-8",
    )
    (repo / "app.py").write_text(
        "from services.auth import authenticate\n\n"
        "def login(name):\n"
        "    return authenticate(name)\n",
        encoding="utf-8",
    )
    (repo / "runner.py").write_text(
        "from app import login\n\ndef main():\n    return login('world')\n",
        encoding="utf-8",
    )
    (repo / "orphan.py").write_text(
        "def standalone():\n    return 42\n",
        encoding="utf-8",
    )


class ResolveGraphTests(unittest.TestCase):
    def test_resolves_file_import_edges(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                counts = resolve_graph(conn, stats.repo_id)
                self.assertGreater(counts["imports"], 0)

    def test_resolves_cross_file_calls_via_imports(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                target = find_symbol(conn, "greet")[0]
                neighbors_out = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND dst = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, target["id"]),
                ).fetchone()[0]
                self.assertGreaterEqual(neighbors_out, 1)

    def test_resolves_extends_to_local_class(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "base.py").write_text(
                "class Animal:\n    def speak(self):\n        pass\n",
                encoding="utf-8",
            )
            (repo / "dog.py").write_text(
                "from base import Animal\n\n"
                "class Dog(Animal):\n    def speak(self):\n        return 'woof'\n",
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                dog_class = find_symbol(conn, "Dog")[0]
                extends = conn.execute(
                    "SELECT e.detail, n.kind FROM edges e "
                    "JOIN nodes n ON n.repo_id = e.repo_id AND n.id = e.dst "
                    "WHERE e.repo_id = ? AND e.src = ? AND e.kind = 'EXTENDS' "
                    "AND e.confidence = 'resolved'",
                    (stats.repo_id, dog_class["id"]),
                ).fetchone()
                self.assertIsNotNone(extends)
                self.assertEqual(extends["kind"], "Class")

    def test_relative_imports_resolve(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            pkg = repo / "pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (pkg / "a.py").write_text("def fn_a():\n    pass\n", encoding="utf-8")
            (pkg / "b.py").write_text(
                "from .a import fn_a\n\ndef fn_b():\n    return fn_a()\n",
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                fn_a = find_symbol(conn, "fn_a")[0]
                resolved = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND dst = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, fn_a["id"]),
                ).fetchone()[0]
                self.assertGreaterEqual(resolved, 1)

    def test_star_import_resolution(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "lib.py").write_text("def exported_func():\n    return 1\n", encoding="utf-8")
            (repo / "consumer.py").write_text(
                "from lib import *\n\ndef caller():\n    return exported_func()\n",
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                fn = find_symbol(conn, "exported_func")[0]
                resolved = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND dst = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, fn["id"]),
                ).fetchone()[0]
                self.assertGreaterEqual(resolved, 1)

    def test_ts_js_named_import_calls_resolve(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "lib.ts").write_text("export function fn() { return 1; }\n", encoding="utf-8")
            (repo / "app.ts").write_text(
                'import { fn } from "./lib";\nexport function main() { return fn(); }\n',
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                fn = find_symbol(conn, "fn")[0]
                resolved = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND dst = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, fn["id"]),
                ).fetchone()[0]
                self.assertEqual(resolved, 1)

    def test_python_import_syntaxes_resolve_calls(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            pkg = repo / "pkg"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("", encoding="utf-8")
            (repo / "lib.py").write_text("def fn():\n    return 1\n", encoding="utf-8")
            (pkg / "mod.py").write_text("def inner():\n    return 2\n", encoding="utf-8")
            (pkg / "mod2.py").write_text("def named():\n    return 3\n", encoding="utf-8")
            (repo / "consumer.py").write_text(
                "import lib\n"
                "import lib as l\n"
                "from pkg import mod\n"
                "from pkg.mod2 import named as alias\n\n"
                "def caller():\n"
                "    lib.fn()\n"
                "    l.fn()\n"
                "    mod.inner()\n"
                "    alias()\n",
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                expected = {"fn": 2, "inner": 1, "named": 1}
                for name, minimum in expected.items():
                    target = find_symbol(conn, name)[0]
                    resolved = conn.execute(
                        "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND dst = ? "
                        "AND kind = 'CALLS' AND confidence = 'resolved'",
                        (stats.repo_id, target["id"]),
                    ).fetchone()[0]
                    self.assertGreaterEqual(resolved, minimum, name)

    def test_imports_disambiguate_duplicate_symbol_names(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "a.py").write_text("def render():\n    return 'a'\n", encoding="utf-8")
            (repo / "b.py").write_text("def render():\n    return 'b'\n", encoding="utf-8")
            (repo / "consumer.py").write_text(
                "from a import render\n\ndef caller():\n    return render()\n",
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                a_render = next(
                    item for item in find_symbol(conn, "render") if item["path"] == "a.py"
                )
                b_render = next(
                    item for item in find_symbol(conn, "render") if item["path"] == "b.py"
                )
                a_edges = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND dst = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, a_render["id"]),
                ).fetchone()[0]
                b_edges = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND dst = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, b_render["id"]),
                ).fetchone()[0]
                self.assertEqual(a_edges, 1)
                self.assertEqual(b_edges, 0)

    def test_self_method_resolves_to_containing_class(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "models.py").write_text(
                "class A:\n"
                "    def save(self):\n"
                "        return 'a'\n"
                "    def run(self):\n"
                "        return self.save()\n\n"
                "class B:\n"
                "    def save(self):\n"
                "        return 'b'\n",
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                run = next(
                    item for item in find_symbol(conn, "run") if item["qname"] == "models.A.run"
                )
                a_save = next(
                    item for item in find_symbol(conn, "save") if item["qname"] == "models.A.save"
                )
                b_save = next(
                    item for item in find_symbol(conn, "save") if item["qname"] == "models.B.save"
                )
                a_edges = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND src = ? AND dst = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, run["id"], a_save["id"]),
                ).fetchone()[0]
                b_edges = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND src = ? AND dst = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, run["id"], b_save["id"]),
                ).fetchone()[0]
                self.assertEqual(a_edges, 1)
                self.assertEqual(b_edges, 0)

    def test_duplicate_methods_do_not_resolve_by_bare_name(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "models.py").write_text(
                "class A:\n"
                "    def save(self):\n"
                "        return 'a'\n\n"
                "class B:\n"
                "    def save(self):\n"
                "        return 'b'\n\n"
                "def caller(obj):\n"
                "    return obj.save()\n",
                encoding="utf-8",
            )
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                caller = find_symbol(conn, "caller")[0]
                resolved = conn.execute(
                    "SELECT COUNT(*) FROM edges WHERE repo_id = ? AND src = ? "
                    "AND kind = 'CALLS' AND confidence = 'resolved'",
                    (stats.repo_id, caller["id"]),
                ).fetchone()[0]
                self.assertEqual(resolved, 0)


class BlastRadiusTests(unittest.TestCase):
    def test_blast_radius_traverses_upstream(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                greet = find_symbol(conn, "greet")[0]
                result = blast_radius(conn, greet["id"], depth=3, direction="up")
                upstream = result.get("upstream", [])
                qnames = {entry["node"]["qname"] for entry in upstream}
                self.assertIn("services.auth.authenticate", qnames)

    def test_blast_radius_traverses_downstream(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                main_fn = find_symbol(conn, "main")[0]
                result = blast_radius(conn, main_fn["id"], depth=3, direction="down")
                downstream = result.get("downstream", [])
                qnames = {entry["node"]["qname"] for entry in downstream}
                self.assertIn("app.login", qnames)

    def test_blast_radius_respects_depth(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                greet = find_symbol(conn, "greet")[0]
                r1 = blast_radius(conn, greet["id"], depth=1, direction="up")
                r3 = blast_radius(conn, greet["id"], depth=3, direction="up")
                self.assertLessEqual(len(r1.get("upstream", [])), len(r3.get("upstream", [])))

    def test_blast_radius_defaults_to_full_reachable_traversal(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                greet = find_symbol(conn, "greet")[0]
                result = blast_radius(conn, greet["id"], direction="up")
                qnames = {entry["node"]["qname"] for entry in result["upstream"]}
                self.assertEqual(result["depth"], None)
                self.assertEqual(result["depth_label"], "all")
                self.assertIn("runner.main", qnames)

    def test_blast_radius_finds_entrypoints(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "app.py").write_text(
                "from fastapi import FastAPI\n\napp = FastAPI()\n\n"
                "def process(data):\n    return data\n\n"
                "@app.post('/process')\ndef handle_process(data: str):\n    return process(data)\n",
                encoding="utf-8",
            )
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                process_fn = find_symbol(conn, "process")[0]
                result = blast_radius(conn, process_fn["id"], depth=2, direction="up")
                entrypoints = result.get("entrypoints", [])
                self.assertTrue(
                    any(ep["node"]["kind"] == "Route" for ep in entrypoints),
                    result,
                )

    def test_blast_radius_includes_files(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                greet = find_symbol(conn, "greet")[0]
                result = blast_radius(conn, greet["id"], depth=3)
                files = [f["path"] for f in result.get("files", [])]
                self.assertIn("models/user.py", files)

    def test_blast_radius_max_nodes_truncates(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                greet = find_symbol(conn, "greet")[0]
                result = blast_radius(conn, greet["id"], depth=5, max_nodes=1, direction="up")
                self.assertTrue(result.get("truncated", False))

    def test_blast_radius_on_nonexistent_node(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                result = blast_radius(conn, "nonexistent_id", depth=3)
                self.assertEqual(result, {})

    def test_blast_radius_labels_file_import_expansion(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                greet = find_symbol(conn, "greet")[0]
                result = blast_radius(conn, greet["id"], direction="up")
                entries = result.get("upstream", [])
                self.assertTrue(
                    any(
                        entry["via"] == "IMPORTS" and entry["scope"] == "conservative_file"
                        for entry in entries
                    ),
                    entries,
                )


class ImpactReportTests(unittest.TestCase):
    def test_impact_report_includes_blast_radius(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                greet = find_symbol(conn, "greet")[0]
                report = impact_report(conn, greet, depth=3, direction="both")
                self.assertIn("blast_radius", report)
                self.assertIn("summary", report)
                self.assertGreater(report["summary"]["upstream"], 0)
                self.assertIn("models/user.py", report["files"])

    def test_impact_targets_by_name(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                targets = impact_targets(conn, "greet", repo_id=stats.repo_id, limit=5)
                self.assertTrue(targets)
                self.assertEqual(targets[0]["name"], "greet")

    def test_impact_targets_by_node_id(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            _write_multi_file_repo(repo)
            stats = index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                greet = find_symbol(conn, "greet")[0]
                targets = impact_targets(conn, greet["id"], repo_id=stats.repo_id, limit=5)
                self.assertEqual(len(targets), 1)
                self.assertEqual(targets[0]["id"], greet["id"])

    def test_impact_report_filters_resolved_external_placeholder(self) -> None:
        with TemporaryDirectory() as repo_tmp, TemporaryDirectory() as data_tmp:
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            (repo / "lib.py").write_text("def fn():\n    return 1\n", encoding="utf-8")
            (repo / "consumer.py").write_text(
                "from lib import fn\n\ndef caller():\n    return fn()\n",
                encoding="utf-8",
            )
            index_repo(repo, sqlite_path(data_dir))

            with closing(connect(sqlite_path(data_dir))) as conn:
                caller = find_symbol(conn, "caller")[0]
                report = impact_report(conn, caller, depth=1, direction="down")
                callee_kinds = {item["node"]["kind"] for item in report["callees"] if item["node"]}
                self.assertIn("Function", callee_kinds)
                self.assertNotIn("ExternalSymbol", callee_kinds)


if __name__ == "__main__":
    unittest.main()
