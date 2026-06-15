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


def write_sample_repo(repo: Path) -> None:
    (repo / "app.py").write_text(APP_CODE, encoding="utf-8")
    (repo / "services.py").write_text(SERVICE_CODE, encoding="utf-8")


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

    def test_operational_benchmark_output_file_matches_stdout(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
            tempfile.TemporaryDirectory() as out_tmp,
        ):
            repo = Path(repo_tmp)
            write_sample_repo(repo)
            output_file = Path(out_tmp) / "output.json"
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
                    "--output",
                    str(output_file),
                ],
                check=True,
                capture_output=True,
                cwd=PROJECT_ROOT,
                text=True,
                timeout=120,
            )
            stdout_payload = json.loads(result.stdout)
            self.assertTrue(output_file.exists())
            file_payload = json.loads(output_file.read_text())
            self.assertEqual(file_payload["ok"], stdout_payload["ok"])
            self.assertEqual(file_payload["repo"], stdout_payload["repo"])
            self.assertEqual(file_payload["cold_index"], stdout_payload["cold_index"])
            self.assertEqual(file_payload["hot_index"], stdout_payload["hot_index"])
            self.assertEqual(file_payload["database"], stdout_payload["database"])

    def test_operational_benchmark_data_dir_explicit(self) -> None:
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
            self.assertEqual(Path(payload["data_dir"]).resolve(), Path(data_tmp).resolve())
            # Ensure data directory was actually used
            self.assertTrue((Path(data_tmp) / "lode.sqlite").exists())

    def test_operational_benchmark_temp_data_dir_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as repo_tmp:
            repo = Path(repo_tmp)
            write_sample_repo(repo)
            result = subprocess.run(
                [
                    sys.executable,
                    "scripts/bench_lode.py",
                    "--repo",
                    str(repo),
                    "--repeat",
                    "2",
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
        data_dir = Path(payload["data_dir"])
        # Temp dir should be cleaned up by default
        self.assertFalse(data_dir.exists())

    def test_operational_benchmark_invalid_repo_machine_readable(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "scripts/bench_lode.py",
                "--repo",
                "/tmp/does_not_exist_12345",
                "--json",
            ],
            capture_output=True,
            cwd=PROJECT_ROOT,
            text=True,
            timeout=30,
        )
        self.assertNotEqual(result.returncode, 0)
        payload = json.loads(result.stdout)
        self.assertFalse(payload["ok"])
        self.assertIn("error", payload)
        self.assertTrue(len(payload["error"]) > 0)
        self.assertIn("does not exist", payload["error"].lower())
        # No traceback in stdout or stderr
        self.assertNotIn("Traceback", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_operational_benchmark_timing_summary_complete(self) -> None:
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
                    "3",
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
        # Check all timing summaries have required fields and ordering
        for section, key in [
            ("search", "create user"),
            ("symbols", "create_user"),
            ("context", "create user"),
        ]:
            summary = payload[section][key]["timing_ms"]
            required_keys = ["count", "min", "p50", "mean", "p95", "max"]
            for rk in required_keys:
                self.assertIn(rk, summary, f"{section}.{key}.timing_ms missing {rk}")
                self.assertIsInstance(
                    summary[rk], (int, float), f"{section}.{key}.timing_ms.{rk} is not numeric"
                )
            self.assertEqual(summary["count"], 3)
            self.assertTrue(
                0 <= summary["min"] <= summary["p50"] <= summary["p95"] <= summary["max"],
                f"{section}.{key} timing order violation: {summary}",
            )

    def test_operational_benchmark_json_shape(self) -> None:
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
                    "--include-kuzu",
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
        # Required top-level keys
        for key in [
            "ok",
            "repo",
            "data_dir",
            "cold_index",
            "hot_index",
            "database",
            "search",
            "symbols",
            "context",
            "neighbors",
            "kuzu",
        ]:
            self.assertIn(key, payload)
        # Cold index stats
        cold_stats = payload["cold_index"]["stats"]
        self.assertGreater(cold_stats["scanned"], 0)
        self.assertGreater(cold_stats["indexed"], 0)
        self.assertGreater(cold_stats["nodes"], 0)
        self.assertGreater(cold_stats["edges"], 0)
        self.assertEqual(cold_stats["removed"], 0)
        self.assertEqual(cold_stats["skipped_unchanged"], 0)
        # Hot index stats
        hot_stats = payload["hot_index"]["stats"]
        self.assertEqual(hot_stats["scanned"], cold_stats["scanned"])
        self.assertEqual(hot_stats["skipped_unchanged"], cold_stats["scanned"])
        self.assertEqual(hot_stats["indexed"], 0)
        self.assertEqual(hot_stats["nodes"], 0)
        self.assertEqual(hot_stats["edges"], 0)
        self.assertEqual(hot_stats["removed"], 0)
        # Database counts
        counts = payload["database"]["counts"]
        self.assertEqual(counts["repos"], 1)
        self.assertGreater(counts["files"], 0)
        self.assertGreater(counts["nodes"], 0)
        self.assertGreater(counts["edges"], 0)
        self.assertGreater(payload["database"]["sqlite_bytes"], 0)
        # Kuzu
        kuzu = payload["kuzu"]
        self.assertGreater(kuzu["timing_ms"], 0)
        self.assertEqual(kuzu["nodes"], counts["nodes"])
        self.assertEqual(kuzu["edges"], counts["edges"])
        # Neighbors
        neighbors = payload["neighbors"]
        self.assertIn("node_id", neighbors)
        self.assertIn("timing_ms", neighbors)
        self.assertTrue(neighbors["incoming"] + neighbors["outgoing"] > 0)

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

    def test_repobench_adapter_full_diagnostics_schema(self) -> None:
        with tempfile.TemporaryDirectory() as data_tmp:
            root = Path(data_tmp)
            first = root / "cross_file_first.jsonl"
            random = root / "cross_file_random.jsonl"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": "tianyang/repobench_python_v1.1",
                        "jsonl_files": [str(first), str(random)],
                    }
                ),
                encoding="utf-8",
            )
            sample = {
                "idx": 0,
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
                "level": "2k",
            }
            first.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            random.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/repobench_adapter.py",
                    "--input",
                    str(root),
                    "--mode",
                    "context",
                    "--top-k",
                    "1",
                    "3",
                    "5",
                    "10",
                    "--query-lines",
                    "5",
                    "--search-limit",
                    "30",
                    "--context-budget",
                    "6000",
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
        # Top-level diagnostics schema
        self.assertEqual(payload["dataset"], "tianyang/repobench_python_v1.1")
        self.assertEqual(
            payload["input_files"], ["cross_file_first.jsonl", "cross_file_random.jsonl"]
        )
        self.assertEqual(payload["splits"], ["cross_file_first", "cross_file_random"])
        self.assertEqual(payload["samples_evaluated"], 2)
        self.assertEqual(payload["samples_skipped"], 0)
        self.assertIn("hit_counts", payload)
        self.assertIn("reciprocal_rank_sum", payload)
        # Split results
        self.assertIn("split_results", payload)
        for split_name in ["cross_file_first", "cross_file_random"]:
            split = payload["split_results"][split_name]
            self.assertEqual(split["split"], split_name)
            self.assertEqual(split["dataset"], "tianyang/repobench_python_v1.1")
            self.assertEqual(split["samples_evaluated"], 1)
            self.assertEqual(split["samples_skipped"], 0)
            self.assertIn("hit_counts", split)
            self.assertIn("reciprocal_rank_sum", split)
            # Buckets
            self.assertIn("by_bucket", split)
            for bucket in ["lt5_candidates", "easy_5_9_candidates", "hard_10_plus_candidates"]:
                self.assertIn(bucket, split["by_bucket"])
                bucket_data = split["by_bucket"][bucket]
                self.assertIn("samples_evaluated", bucket_data)
                self.assertIn("metrics", bucket_data)
                self.assertIn("hit_counts", bucket_data)
                self.assertIn("reciprocal_rank_sum", bucket_data)
            # Levels
            self.assertIn("by_level", split)
            self.assertIn("2k", split["by_level"])
            level_data = split["by_level"]["2k"]
            self.assertEqual(level_data["samples_evaluated"], 1)
            self.assertIn("metrics", level_data)
            self.assertIn("hit_counts", level_data)
            self.assertIn("reciprocal_rank_sum", level_data)

    def test_repobench_adapter_exact_skip_identities(self) -> None:
        with tempfile.TemporaryDirectory() as data_tmp:
            input_path = Path(data_tmp) / "repobench.jsonl"
            # Sample with invalid gold_snippet_index
            sample = {
                "idx": 0,
                "repo_name": "synthetic",
                "file_path": "app.py",
                "cropped_code": APP_CODE.rstrip("\n"),
                "context": [
                    {
                        "identifier": "UserService",
                        "path": "services.py",
                        "snippet": SERVICE_CODE.rstrip("\n"),
                    },
                ],
                "gold_snippet_index": 5,  # invalid: only 1 context item
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
        self.assertEqual(payload["samples_evaluated"], 0)
        self.assertEqual(payload["samples_skipped"], 1)
        # Error identity
        self.assertEqual(len(payload["errors"]), 1)
        error = payload["errors"][0]
        self.assertEqual(error["sample_id"], "repobench.jsonl:1")
        self.assertEqual(error["line"], 1)
        self.assertIn("gold_snippet_index is outside context list", error["error"])
        # Split-level error
        split = payload["split_results"]["repobench"]
        self.assertEqual(split["samples_skipped"], 1)
        self.assertEqual(len(split["errors"]), 1)
        self.assertEqual(split["errors"][0]["sample_id"], "repobench.jsonl:1")
        self.assertEqual(split["errors"][0]["line"], 1)

    def test_repobench_adapter_skip_accounted_in_bucket_and_level(self) -> None:
        with tempfile.TemporaryDirectory() as data_tmp:
            root = Path(data_tmp)
            first = root / "cross_file_first.jsonl"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": "tianyang/repobench_python_v1.1",
                        "jsonl_files": [str(first)],
                    }
                ),
                encoding="utf-8",
            )
            valid = {
                "idx": 0,
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
                "level": "2k",
            }
            invalid = {
                "idx": 1,
                "repo_name": "synthetic",
                "file_path": "app.py",
                "cropped_code": APP_CODE.rstrip("\n"),
                "context": [
                    {
                        "identifier": "UserService",
                        "path": "services.py",
                        "snippet": SERVICE_CODE.rstrip("\n"),
                    },
                ],
                "gold_snippet_index": 5,
                "next_line": "    return service.save_user(name)",
                "level": "4k",
            }
            first.write_text(
                json.dumps(valid) + "\n" + json.dumps(invalid) + "\n", encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/repobench_adapter.py",
                    "--input",
                    str(root),
                    "--mode",
                    "context",
                    "--top-k",
                    "1",
                    "3",
                    "5",
                    "10",
                    "--query-lines",
                    "5",
                    "--search-limit",
                    "30",
                    "--context-budget",
                    "6000",
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
        self.assertEqual(payload["samples_skipped"], 1)
        split = payload["split_results"]["cross_file_first"]
        self.assertEqual(split["samples_evaluated"], 1)
        self.assertEqual(split["samples_skipped"], 1)
        # Valid sample has 2 contexts -> lt5_candidates, level 2k
        # Invalid sample has 1 context -> lt5_candidates, level 4k
        by_bucket = split["by_bucket"]
        self.assertEqual(by_bucket["lt5_candidates"]["samples_evaluated"], 1)
        self.assertEqual(by_bucket["lt5_candidates"]["samples_skipped"], 1)
        self.assertEqual(by_bucket["easy_5_9_candidates"]["samples_evaluated"], 0)
        self.assertEqual(by_bucket["easy_5_9_candidates"]["samples_skipped"], 0)
        self.assertEqual(by_bucket["hard_10_plus_candidates"]["samples_evaluated"], 0)
        self.assertEqual(by_bucket["hard_10_plus_candidates"]["samples_skipped"], 0)
        by_level = split["by_level"]
        self.assertEqual(by_level["2k"]["samples_evaluated"], 1)
        self.assertEqual(by_level["2k"]["samples_skipped"], 0)
        self.assertEqual(by_level["4k"]["samples_evaluated"], 0)
        self.assertEqual(by_level["4k"]["samples_skipped"], 1)
        # Errors recorded at all levels
        self.assertEqual(len(by_bucket["lt5_candidates"]["errors"]), 1)
        self.assertEqual(len(by_level["4k"]["errors"]), 1)

    def test_repobench_adapter_directory_limit_is_global(self) -> None:
        with tempfile.TemporaryDirectory() as data_tmp:
            root = Path(data_tmp)
            first = root / "cross_file_first.jsonl"
            random = root / "cross_file_random.jsonl"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": "tianyang/repobench_python_v1.1",
                        "jsonl_files": [str(first), str(random)],
                    }
                ),
                encoding="utf-8",
            )
            sample = {
                "idx": 0,
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
                "level": "2k",
            }
            first.write_text(
                json.dumps(sample) + "\n" + json.dumps(sample) + "\n", encoding="utf-8"
            )
            random.write_text(
                json.dumps(sample) + "\n" + json.dumps(sample) + "\n", encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/repobench_adapter.py",
                    "--input",
                    str(root),
                    "--mode",
                    "context",
                    "--top-k",
                    "1",
                    "3",
                    "5",
                    "10",
                    "--query-lines",
                    "5",
                    "--search-limit",
                    "30",
                    "--context-budget",
                    "6000",
                    "--limit",
                    "1",
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
        # Only one split should have evaluated a sample; the other is skipped by global limit
        total_evaluated = sum(s["samples_evaluated"] for s in payload["split_results"].values())
        self.assertEqual(total_evaluated, 1)

    def test_repobench_adapter_details_bounded_by_limit(self) -> None:
        with tempfile.TemporaryDirectory() as data_tmp:
            root = Path(data_tmp)
            first = root / "cross_file_first.jsonl"
            random = root / "cross_file_random.jsonl"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": "tianyang/repobench_python_v1.1",
                        "jsonl_files": [str(first), str(random)],
                    }
                ),
                encoding="utf-8",
            )
            sample = {
                "idx": 0,
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
                "level": "2k",
            }
            first.write_text(
                json.dumps(sample) + "\n" + json.dumps(sample) + "\n", encoding="utf-8"
            )
            random.write_text(
                json.dumps(sample) + "\n" + json.dumps(sample) + "\n", encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/repobench_adapter.py",
                    "--input",
                    str(root),
                    "--mode",
                    "context",
                    "--top-k",
                    "1",
                    "3",
                    "5",
                    "10",
                    "--query-lines",
                    "5",
                    "--search-limit",
                    "30",
                    "--context-budget",
                    "6000",
                    "--limit",
                    "1",
                    "--details",
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
        self.assertEqual(len(payload["details"]), 1)

    def test_repobench_adapter_stops_parsing_after_limit(self) -> None:
        """When --limit is reached, the adapter stops before reading any post-cap row,
        even malformed JSON in the same split file."""
        with tempfile.TemporaryDirectory() as data_tmp:
            root = Path(data_tmp)
            first = root / "cross_file_first.jsonl"
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "dataset": "tianyang/repobench_python_v1.1",
                        "jsonl_files": [str(first)],
                    }
                ),
                encoding="utf-8",
            )
            sample = {
                "idx": 0,
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
                "level": "2k",
            }
            # One valid sample, then malformed JSON immediately after the cap
            first.write_text(json.dumps(sample) + "\nthis is not valid json\n", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    "benchmarks/repobench_adapter.py",
                    "--input",
                    str(root),
                    "--mode",
                    "context",
                    "--top-k",
                    "1",
                    "3",
                    "5",
                    "10",
                    "--query-lines",
                    "5",
                    "--search-limit",
                    "30",
                    "--context-budget",
                    "6000",
                    "--limit",
                    "1",
                    "--json",
                ],
                capture_output=True,
                cwd=PROJECT_ROOT,
                text=True,
                timeout=120,
            )
        payload = json.loads(result.stdout)
        # Must succeed because the adapter stops before reading the malformed JSON
        self.assertTrue(payload["ok"], msg=result.stdout + result.stderr)
        self.assertEqual(payload["samples_evaluated"], 1)


if __name__ == "__main__":
    unittest.main()
