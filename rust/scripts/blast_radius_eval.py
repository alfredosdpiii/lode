#!/usr/bin/env python3
"""Precision/recall evaluation harness for Lode's call resolution and blast_radius.

Runs the Rust `lode` binary on:
1. A ground-truth synthetic benchmark suite covering receiver resolution,
   false-positive rejection, and import-aware blast radius filtering.
2. A real-repo audit on Lode's own `src/lode` codebase.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BIN = ROOT / "rust" / "target" / "debug" / "lode"


def run_lode(binary: Path, data_dir: Path, *args: str) -> dict:
    proc = subprocess.run(
        [str(binary), "--data-dir", str(data_dir), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def write_files(root: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def resolved_call_pairs(db_path: Path) -> set[tuple[str, str]]:
    """Return (caller_qname, callee_qname) for all non-external CALLS edges."""
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        """
        SELECT src.qname, dst.qname
        FROM edges e
        JOIN nodes src ON src.repo_id = e.repo_id AND src.id = e.src
        JOIN nodes dst ON dst.repo_id = e.repo_id AND dst.id = e.dst
        WHERE e.kind = 'CALLS' AND dst.kind NOT IN ('ExternalSymbol', 'ExternalDependency')
        """
    ).fetchall()
    conn.close()
    return {(r[0], r[1]) for r in rows}


def eval_ground_truth(binary: Path) -> dict:
    cases = [
        {
            "name": "constructor_disambiguation",
            "files": {
                "a.py": "class A:\n    def save(self):\n        return 'a'\n",
                "b.py": "class B:\n    def save(self):\n        return 'b'\n",
                "main.py": (
                    "from a import A\nfrom b import B\n\n"
                    "def run():\n    svc = A()\n    return svc.save()\n"
                ),
            },
            "expected_calls": {("main.run", "a.A.save")},
            "forbidden_calls": {("main.run", "b.B.save")},
        },
        {
            "name": "param_annotation_disambiguation",
            "files": {
                "users.py": "class UserService:\n    def save(self):\n        return 1\n",
                "orders.py": "class OrderService:\n    def save(self):\n        return 2\n",
                "handler.py": (
                    "from users import UserService\nfrom orders import OrderService\n\n"
                    "def handle(svc: UserService):\n    return svc.save()\n"
                ),
            },
            "expected_calls": {("handler.handle", "users.UserService.save")},
            "forbidden_calls": {("handler.handle", "orders.OrderService.save")},
        },
        {
            "name": "optional_union_annotation",
            "files": {
                "users.py": "class UserService:\n    def save(self):\n        return 1\n",
                "orders.py": "class OrderService:\n    def save(self):\n        return 2\n",
                "handler.py": (
                    "from typing import Optional\n"
                    "from users import UserService\n"
                    "from orders import OrderService\n\n"
                    "def f1(svc: Optional[UserService]):\n    return svc.save()\n\n"
                    "def f2(svc: OrderService | None):\n    return svc.save()\n"
                ),
            },
            "expected_calls": {
                ("handler.f1", "users.UserService.save"),
                ("handler.f2", "orders.OrderService.save"),
            },
            "forbidden_calls": {
                ("handler.f1", "orders.OrderService.save"),
                ("handler.f2", "users.UserService.save"),
            },
        },
        {
            "name": "ann_assign_receiver",
            "files": {
                "users.py": "class UserService:\n    def save(self):\n        return 1\n",
                "orders.py": "class OrderService:\n    def save(self):\n        return 2\n",
                "handler.py": (
                    "from users import UserService\n\n"
                    "def handle(factory):\n"
                    "    svc: UserService = factory()\n"
                    "    return svc.save()\n"
                ),
            },
            "expected_calls": {("handler.handle", "users.UserService.save")},
            "forbidden_calls": {("handler.handle", "orders.OrderService.save")},
        },
        {
            "name": "self_attr_init_binding",
            "files": {
                "users.py": "class UserService:\n    def save(self):\n        return 1\n",
                "orders.py": "class OrderService:\n    def save(self):\n        return 2\n",
                "ctrl.py": (
                    "from users import UserService\n\n"
                    "class Controller:\n"
                    "    def __init__(self):\n"
                    "        self.svc = UserService()\n"
                    "    def handle(self):\n"
                    "        return self.svc.save()\n"
                ),
            },
            "expected_calls": {("ctrl.Controller.handle", "users.UserService.save")},
            "forbidden_calls": {("ctrl.Controller.handle", "orders.OrderService.save")},
        },
        {
            "name": "inherited_method_call",
            "files": {
                "base.py": (
                    "class Base:\n    def execute(self):\n        return 1\n"
                ),
                "other.py": (
                    "class Other:\n    def execute(self):\n        return 2\n"
                ),
                "sub.py": (
                    "from base import Base\n\n"
                    "class Sub(Base):\n"
                    "    def run(self):\n"
                    "        return self.execute()\n\n"
                    "def entry():\n"
                    "    s = Sub()\n"
                    "    return s.execute()\n"
                ),
            },
            "expected_calls": {
                ("sub.Sub.run", "base.Base.execute"),
                ("sub.entry", "base.Base.execute"),
            },
            "forbidden_calls": {
                ("sub.Sub.run", "other.Other.execute"),
                ("sub.entry", "other.Other.execute"),
            },
        },
        {
            "name": "with_context_manager",
            "files": {
                "client.py": "class Client:\n    def fetch(self):\n        return 1\n",
                "other.py": "class Other:\n    def fetch(self):\n        return 2\n",
                "main.py": (
                    "from client import Client\n\n"
                    "def run():\n"
                    "    with Client() as c:\n"
                    "        return c.fetch()\n"
                ),
            },
            "expected_calls": {("main.run", "client.Client.fetch")},
            "forbidden_calls": {("main.run", "other.Other.fetch")},
        },
        {
            "name": "module_level_instance",
            "files": {
                "client.py": "class Client:\n    def fetch(self):\n        return 1\n",
                "other.py": "class Other:\n    def fetch(self):\n        return 2\n",
                "main.py": (
                    "from client import Client\n\n"
                    "client = Client()\n\n"
                    "def run():\n"
                    "    return client.fetch()\n"
                ),
            },
            "expected_calls": {("main.run", "client.Client.fetch")},
            "forbidden_calls": {("main.run", "other.Other.fetch")},
        },
        {
            "name": "direct_ctor_chain",
            "files": {
                "users.py": "class UserService:\n    def create(self):\n        return 1\n",
                "orders.py": "class OrderService:\n    def create(self):\n        return 2\n",
                "main.py": (
                    "from users import UserService\n\n"
                    "def run():\n"
                    "    return UserService().create()\n"
                ),
            },
            "expected_calls": {("main.run", "users.UserService.create")},
            "forbidden_calls": {("main.run", "orders.OrderService.create")},
        },
        {
            "name": "false_positive_third_party_method",
            "files": {
                "store.py": "class LocalStore:\n    def execute(self):\n        return 1\n",
                "db.py": (
                    "import sqlite3\n\n"
                    "def query(conn):\n"
                    "    return conn.execute('SELECT 1')\n"
                ),
            },
            "expected_calls": set(),
            "forbidden_calls": {("db.query", "store.LocalStore.execute")},
        },
        {
            "name": "false_positive_unknown_receiver",
            "files": {
                "buf.py": "class Buffer:\n    def flush(self):\n        return 1\n",
                "io_mod.py": (
                    "def write_out(stream):\n"
                    "    stream.flush()\n"
                ),
            },
            "expected_calls": set(),
            "forbidden_calls": {("io_mod.write_out", "buf.Buffer.flush")},
        },
        {
            "name": "selective_import_blast_radius",
            "files": {
                "utils.py": (
                    "def helper_a():\n    return 'a'\n\n"
                    "def helper_b():\n    return 'b'\n"
                ),
                "uses_a.py": (
                    "from utils import helper_a\n\n"
                    "def run_a():\n    return helper_a()\n"
                ),
                "uses_b.py": (
                    "from utils import helper_b\n\n"
                    "def run_b():\n    return helper_b()\n"
                ),
                "uses_mod.py": (
                    "import utils\n\n"
                    "def run_mod():\n    return utils.helper_b()\n"
                ),
                "uses_star.py": (
                    "from utils import *\n\n"
                    "def run_star():\n    return helper_b()\n"
                ),
            },
            "blast_target": "helper_b",
            "expected_blast_files": {"utils.py", "uses_b.py", "uses_mod.py", "uses_star.py"},
            "forbidden_blast_files": {"uses_a.py"},
        },
    ]

    tp = fp = fn = 0
    blast_tp = blast_fp = blast_fn = 0
    details = []

    for case in cases:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            repo = tmp_path / "repo"
            data = tmp_path / "data"
            write_files(repo, case["files"])
            run_lode(binary, data, "index", str(repo), "--json")

            entry: dict = {"name": case["name"], "passed": True}
            if "expected_calls" in case:
                actual = resolved_call_pairs(data / "lode.sqlite")
                exp = case["expected_calls"]
                forb = case["forbidden_calls"]
                hit = len(exp & actual)
                miss = len(exp - actual)
                wrong = len(forb & actual)
                tp += hit
                fn += miss
                fp += wrong
                entry["call_tp"] = hit
                entry["call_fn"] = miss
                entry["call_fp"] = wrong
                if miss > 0 or wrong > 0:
                    entry["passed"] = False
                    entry["missing"] = sorted(exp - actual)
                    entry["spurious"] = sorted(forb & actual)

            if "blast_target" in case:
                impact = run_lode(
                    binary, data, "impact", case["blast_target"], "--repo", str(repo), "--json"
                )
                files = set(impact["results"][0]["files"])
                exp_f = case["expected_blast_files"]
                forb_f = case["forbidden_blast_files"]
                hit = len(exp_f & files)
                miss = len(exp_f - files)
                wrong = len(forb_f & files)
                blast_tp += hit
                blast_fn += miss
                blast_fp += wrong
                entry["blast_tp"] = hit
                entry["blast_fn"] = miss
                entry["blast_fp"] = wrong
                if miss > 0 or wrong > 0:
                    entry["passed"] = False
                    entry["missing_files"] = sorted(exp_f - files)
                    entry["spurious_files"] = sorted(forb_f & files)

            details.append(entry)

    def prf(t: int, f_p: int, f_n: int) -> dict:
        prec = t / (t + f_p) if (t + f_p) else 1.0
        rec = t / (t + f_n) if (t + f_n) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"tp": t, "fp": f_p, "fn": f_n, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)}

    return {
        "cases_passed": sum(1 for d in details if d["passed"]),
        "cases_total": len(details),
        "calls": prf(tp, fp, fn),
        "blast_files": prf(blast_tp, blast_fp, blast_fn),
        "details": details,
    }


def eval_real_repo(binary: Path, repo: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        stats = run_lode(binary, data, "index", str(repo), "--json")
        db = sqlite3.connect(str(data / "lode.sqlite"))

        total_call_edges = db.execute("SELECT COUNT(*) FROM edges WHERE kind = 'CALLS'").fetchone()[0]
        resolved_calls = db.execute(
            "SELECT COUNT(*) FROM edges WHERE kind = 'CALLS' AND confidence IN ('strong', 'resolved')"
        ).fetchone()[0]
        unresolved_ext = db.execute(
            """
            SELECT COUNT(*) FROM edges e
            JOIN nodes dst ON dst.repo_id = e.repo_id AND dst.id = e.dst
            WHERE e.kind = 'CALLS' AND dst.kind = 'ExternalSymbol'
            """
        ).fetchone()[0]

        # Spot-check suspicious dotted resolutions where detail's receiver looks like a stdlib/var
        # and the target is a local Function/Method whose owner file isn't imported by caller.
        dotted_resolved = db.execute(
            """
            SELECT src.path, src.qname, e.detail, dst.path, dst.qname, dst.kind
            FROM edges e
            JOIN nodes src ON src.repo_id = e.repo_id AND src.id = e.src
            JOIN nodes dst ON dst.repo_id = e.repo_id AND dst.id = e.dst
            WHERE e.kind = 'CALLS' AND e.confidence = 'resolved' AND e.detail LIKE '%.%'
            """
        ).fetchall()

        # Sample up to 60 Function/Method nodes across the repo for blast radius stats
        symbols = db.execute(
            """
            SELECT id, qname, path FROM nodes
            WHERE kind IN ('Function', 'Method')
            ORDER BY path, start_line
            LIMIT 60
            """
        ).fetchall()
        langs = dict(
            db.execute("SELECT language, COUNT(*) FROM files GROUP BY language").fetchall()
        )
        db.close()

        upstream_counts = []
        file_counts = []
        conservative_counts = []
        for node_id, _qname, _path in symbols:
            rep = run_lode(binary, data, "impact", node_id, "--repo", str(repo), "--json")
            res = rep["results"][0]
            br = res["blast_radius"]
            upstream_counts.append(br["stats"]["upstream"])
            file_counts.append(br["stats"]["files"])
            conservative_counts.append(br["stats"]["conservative_file_expansions"])

        n = max(len(symbols), 1)
        return {
            "repo": str(repo),
            "files_scanned": stats["scanned"],
            "languages": langs,
            "nodes": stats["nodes"],
            "edges": stats["edges"],
            "resolved_imports": stats["resolved_imports"],
            "resolved_calls": stats["resolved_calls"],
            "resolved_extends": stats["resolved_extends"],
            "total_call_edges": total_call_edges,
            "resolved_or_strong_calls": resolved_calls,
            "unresolved_external_calls": unresolved_ext,
            "dotted_resolved_count": len(dotted_resolved),
            "sampled_symbols": len(symbols),
            "mean_upstream_nodes": round(sum(upstream_counts) / n, 2),
            "mean_blast_files": round(sum(file_counts) / n, 2),
            "mean_conservative_file_expansions": round(sum(conservative_counts) / n, 2),
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    gt = eval_ground_truth(args.bin)
    real = eval_real_repo(args.bin, args.repo)
    out = {"ground_truth": gt, "real_repo": real}

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"Ground truth: {gt['cases_passed']}/{gt['cases_total']} cases passed")
        print(f"  Calls:       {gt['calls']}")
        print(f"  Blast files: {gt['blast_files']}")
        for d in gt["details"]:
            if not d["passed"]:
                print(f"  FAIL {d['name']}: {d}")
        print(f"\nReal repo ({real['repo']}):")
        for k, v in real.items():
            if k != "repo":
                print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
