from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

APP_CODE = """from services import UserService

def create_user(name):
    service = UserService()
    return service.save_user(name)
"""

SERVICE_CODE = """class UserService:
    def save_user(self, name):
        return {'name': name}
"""


class BenchmarkScriptTests(unittest.TestCase):
    def test_operational_benchmark_outputs_json_metrics(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            write_sample_repo(repo)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/bench_lode.py",
                    "--repo",
                    str(repo),
                    "--data-dir",
                    data_tmp,
                    "--reset",
                    "--repeat",
                    "2",
                    "--query",
                    "create user",
                    "--symbol",
                    "create_user",
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
        self.assertEqual(payload["cold_index"]["stats"]["scanned"], 2)
        self.assertGreater(payload["database"]["counts"]["nodes"], 0)
        self.assertIn("create user", payload["search"])
        self.assertIn("create_user", payload["symbols"])

    def test_repobench_adapter_scores_synthetic_sample(self) -> None:
        with tempfile.TemporaryDirectory() as data_tmp:
            input_path = Path(data_tmp) / "repobench.jsonl"
            sample = {
                "idx": "synthetic-1",
                "repo_name": "synthetic",
                "file_path": "app.py",
                "cropped_code": APP_CODE.rstrip("\n"),
                "context": [
                    {
                        "identifier": "UserService",
                        "path": "services.py",
                        "snippet": SERVICE_CODE.rstrip("\n"),
                    },
                    {
                        "identifier": "unrelated",
                        "path": "other.py",
                        "snippet": "def unrelated():\n    return None",
                    },
                ],
                "gold_snippet_index": 0,
                "next_line": "    return service.save_user(name)",
            }
            input_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/repobench_adapter.py",
                    "--input",
                    str(input_path),
                    "--top-k",
                    "1",
                    "3",
                    "--query-lines",
                    "5",
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
        self.assertEqual(payload["samples_evaluated"], 1)
        self.assertGreaterEqual(payload["metrics"]["hit_at_3"], 1.0)


def write_sample_repo(repo: Path) -> None:
    (repo / "app.py").write_text(APP_CODE, encoding="utf-8")
    (repo / "services.py").write_text(SERVICE_CODE, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
