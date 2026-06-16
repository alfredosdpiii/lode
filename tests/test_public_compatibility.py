from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from lode.daemon import LodeHandler
from lode.observability import Metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_PORT = 7979

APP_CODE = """from services import UserService

def create_user(name: str):
    service = UserService()
    return service.save_user(name)
"""

SERVICE_CODE = """class UserService:
    def save_user(self, name: str):
        return {"name": name}
"""

PUBLIC_NODE_FIELDS = {
    "id",
    "repo_id",
    "owner_path",
    "kind",
    "name",
    "qname",
    "path",
    "start_line",
    "end_line",
    "signature",
    "doc",
    "confidence",
    "content_hash",
    "rank",
    "extra",
}


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


def write_fixture_repo(repo: Path, service_name: str = "UserService") -> None:
    app_code = APP_CODE.replace("UserService", service_name)
    service_code = SERVICE_CODE.replace("UserService", service_name)
    (repo / "app.py").write_text(app_code, encoding="utf-8")
    (repo / "services.py").write_text(service_code, encoding="utf-8")


def command_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("LODE_DATA_DIR", None)
    env.pop("KG_DATA_DIR", None)
    env.pop("LODE_EMBEDDINGS_URL", None)
    env.pop("KG_EMBEDDINGS_URL", None)
    if extra:
        env.update(extra)
    return env


def run_command(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    check: bool = True,
    timeout: float = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        check=check,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        timeout=timeout,
    )


def run_lode_json(
    data_dir: Path | None,
    *args: str,
    env: dict[str, str] | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "lode"]
    if data_dir is not None:
        command.extend(["--data-dir", str(data_dir)])
    command.extend(args)
    result = run_command(command, env=env, timeout=timeout)
    return json.loads(result.stdout)


def assert_node_schema(test: unittest.TestCase, node: dict[str, Any]) -> None:
    test.assertTrue(PUBLIC_NODE_FIELDS.issubset(node.keys()), node.keys())
    test.assertNotIn("extra_json", node)
    test.assertIsInstance(node["path"], str)
    test.assertFalse(Path(node["path"]).is_absolute(), node["path"])
    test.assertIsInstance(node["start_line"], int)
    test.assertIsInstance(node["end_line"], int)
    test.assertGreaterEqual(node["start_line"], 1)
    test.assertGreaterEqual(node["end_line"], node["start_line"])


def assert_index_stats(test: unittest.TestCase, payload: dict[str, Any], repo: Path) -> None:
    test.assertTrue(payload["ok"])
    test.assertEqual(payload["root"], str(repo.resolve()))
    for field in [
        "repo_id",
        "root",
        "scanned",
        "indexed",
        "skipped_unchanged",
        "removed",
        "resolved_imports",
        "resolved_calls",
        "resolved_extends",
        "nodes",
        "edges",
    ]:
        test.assertIn(field, payload)


def request_json(
    method: str,
    path: str,
    *,
    payload: Any | None = None,
    raw: bytes | None = None,
    headers: dict[str, str] | None = None,
    expected_status: int = 200,
) -> tuple[int, dict[str, str], dict[str, Any]]:
    data = raw
    request_headers = headers.copy() if headers else {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("content-type", "application/json")
    request = urllib.request.Request(
        f"http://127.0.0.1:{VALIDATION_PORT}{path}",
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            status = response.status
            response_headers = dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        status = exc.code
        response_headers = dict(exc.headers.items())
    if status != expected_status:
        raise AssertionError(f"expected HTTP {expected_status}, got {status}: {body}")
    return status, response_headers, json.loads(body)


def request_text(path: str) -> tuple[int, dict[str, str], str]:
    request = urllib.request.Request(f"http://127.0.0.1:{VALIDATION_PORT}{path}", method="GET")
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, dict(response.headers.items()), response.read().decode("utf-8")


def wait_for_health() -> None:
    deadline = time.monotonic() + 10
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request_json("GET", "/health")
            return
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise TimeoutError(f"daemon did not become healthy: {last_error}")


def assert_port_free() -> None:
    deadline = time.monotonic() + 2
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", VALIDATION_PORT))
                return
            except OSError as exc:
                last_error = exc
        time.sleep(0.1)
    if last_error is None:
        raise AssertionError(f"127.0.0.1:{VALIDATION_PORT} must be free")
    raise AssertionError(f"127.0.0.1:{VALIDATION_PORT} must be free") from last_error


def assert_port_released() -> None:
    deadline = time.monotonic() + 5
    last_error: AssertionError | None = None
    while time.monotonic() < deadline:
        try:
            assert_port_free()
            return
        except AssertionError as exc:
            last_error = exc
            time.sleep(0.1)
    if last_error:
        raise last_error


def terminate_process(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)


def normalized_search_tuples(payload: dict[str, Any]) -> list[tuple[str, str, str, int]]:
    return [
        (item["path"], item["kind"], item["qname"], item["start_line"])
        for item in payload["results"]
    ]


def context_paths(payload: dict[str, Any], key: str) -> list[str]:
    return [item["path"] for item in payload[key]]


def first_impact_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["results"][0]["summary"]


def radius_nodes(payload: dict[str, Any]) -> list[dict[str, Any]]:
    radius = payload["results"][0]["blast_radius"]
    nodes: list[dict[str, Any]] = []
    for key in ["upstream", "downstream", "entrypoints"]:
        nodes.extend(entry["node"] for entry in radius.get(key, []))
    return nodes


class PublicCliCompatibilityTests(unittest.TestCase):
    def test_public_entrypoint_help_and_data_dir_environment_precedence(self) -> None:
        help_commands = [
            ["lode", "--help"],
            ["kg", "--help"],
            ["loded", "--help"],
            ["kgd", "--help"],
            [sys.executable, "-m", "lode", "--help"],
            [sys.executable, "-m", "lode.daemon", "--help"],
            ["lode", "serve", "--help"],
        ]
        for command in help_commands:
            with self.subTest(command=command):
                result = run_command(command)
                self.assertEqual(result.returncode, 0)
                self.assertNotIn("Traceback", result.stdout + result.stderr)
                if command[:1] in (["lode"], ["kg"]) or command[:3] == [
                    sys.executable,
                    "-m",
                    "lode",
                ]:
                    stdout = result.stdout
                    if "serve" not in command:
                        self.assertIn("--data-dir", stdout)
                        for subcommand in [
                            "index",
                            "status",
                            "search",
                            "symbol",
                            "context",
                            "impact",
                            "neighbors",
                            "kuzu-sync",
                            "embed",
                            "serve",
                        ]:
                            self.assertIn(subcommand, stdout)
                    else:
                        self.assertIn("--host", stdout)
                        self.assertIn("--port", stdout)
                if command[:1] in (["loded"], ["kgd"]) or command[:3] == [
                    sys.executable,
                    "-m",
                    "lode.daemon",
                ]:
                    self.assertIn("--host", result.stdout)
                    self.assertIn("--port", result.stdout)
                    self.assertIn("--data-dir", result.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            write_fixture_repo(repo)

            lode_data = root / "lode-env-data"
            env = command_env({"LODE_DATA_DIR": str(lode_data)})
            index = run_lode_json(None, "index", str(repo), "--json", env=env)
            assert_index_stats(self, index, repo)
            status = run_lode_json(None, "status", "--json", env=env)
            self.assertEqual(Path(status["data_dir"]).resolve(), lode_data.resolve())
            self.assertEqual(status["repos"][0]["root"], str(repo.resolve()))

            kg_data = root / "kg-env-data"
            env = command_env({"KG_DATA_DIR": str(kg_data)})
            run_lode_json(None, "index", str(repo), "--json", env=env)
            status = run_lode_json(None, "status", "--json", env=env)
            self.assertEqual(Path(status["data_dir"]).resolve(), kg_data.resolve())

            unused_env_data = root / "unused-env-data"
            override_data = root / "override-data"
            env = command_env({"LODE_DATA_DIR": str(unused_env_data)})
            run_lode_json(override_data, "index", str(repo), "--json", env=env)
            status = run_lode_json(override_data, "status", "--json", env=env)
            self.assertEqual(Path(status["data_dir"]).resolve(), override_data.resolve())
            self.assertTrue((override_data / "lode.sqlite").exists())
            self.assertFalse((unused_env_data / "lode.sqlite").exists())

    def test_cli_json_contracts_for_index_query_context_graph_and_errors(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_fixture_repo(repo)

            index = run_lode_json(data_dir, "index", str(repo), "--json")
            assert_index_stats(self, index, repo)
            with sqlite3.connect(data_dir / "lode.sqlite") as conn:
                conn.execute(
                    """
                    UPDATE nodes
                    SET start_line = 0, end_line = 0
                    WHERE kind IN ('ExternalSymbol', 'ExternalDependency')
                    """
                )

            status = run_lode_json(data_dir, "status", "--json")
            self.assertEqual(Path(status["data_dir"]).resolve(), data_dir.resolve())
            repo_entry = status["repos"][0]
            for field in ["id", "root", "name", "indexed_at", "files", "nodes"]:
                self.assertIn(field, repo_entry)

            search = run_lode_json(
                data_dir,
                "search",
                "UserService save_user",
                "--repo",
                str(repo),
                "--limit",
                "5",
                "--json",
            )
            self.assertTrue(search["ok"])
            self.assertLessEqual(len(search["results"]), 5)
            self.assertTrue(search["results"])
            self.assertNotIn(search["results"][0]["kind"], {"ExternalSymbol", "ExternalDependency"})
            for result in search["results"]:
                assert_node_schema(self, result)
                self.assertEqual(result["repo_id"], index["repo_id"])

            stale_external = run_lode_json(
                data_dir,
                "search",
                "services",
                "--repo",
                str(repo),
                "--limit",
                "20",
                "--json",
            )
            self.assertTrue(
                any(
                    result["kind"] in {"ExternalSymbol", "ExternalDependency"}
                    for result in stale_external["results"]
                )
            )
            for result in stale_external["results"]:
                assert_node_schema(self, result)

            symbol = run_lode_json(
                data_dir,
                "symbol",
                "UserService",
                "--repo",
                str(repo),
                "--limit",
                "5",
                "--json",
            )
            self.assertTrue(symbol["ok"])
            self.assertLessEqual(len(symbol["results"]), 5)
            self.assertEqual(symbol["results"][0]["kind"], "Class")
            for result in symbol["results"]:
                assert_node_schema(self, result)

            context = run_lode_json(
                data_dir,
                "context",
                "UserService save_user",
                "--repo",
                str(repo),
                "--budget",
                "2000",
                "--limit",
                "5",
                "--json",
            )
            self.assertTrue(context["ok"])
            self.assertEqual(context["query"], "UserService save_user")
            self.assertEqual(context["budget"], 2000)
            self.assertTrue(context["must_read"])
            self.assertLessEqual(len(context["top_hits"]), 5)
            for field in ["summary", "related", "confidence", "notes"]:
                self.assertIn(field, context)
            for item in context["must_read"]:
                self.assertTrue(
                    {
                        "node_id",
                        "kind",
                        "name",
                        "qname",
                        "path",
                        "start_line",
                        "end_line",
                        "signature",
                        "doc",
                        "why",
                        "confidence",
                    }.issubset(item.keys())
                )
            for item in context["top_hits"]:
                self.assertTrue(
                    {"id", "kind", "qname", "path", "lines", "confidence"}.issubset(item.keys())
                )

            create_symbol = run_lode_json(
                data_dir,
                "symbol",
                "create_user",
                "--repo",
                str(repo),
                "--json",
            )
            node_id = create_symbol["results"][0]["id"]
            neighbors = run_lode_json(data_dir, "neighbors", node_id, "--limit", "5", "--json")
            self.assertTrue(neighbors["ok"])
            self.assertEqual(neighbors["node"]["id"], node_id)
            for neighbor in neighbors["incoming"] + neighbors["outgoing"]:
                self.assertIn("edge", neighbor)
                self.assertTrue({"kind", "confidence", "detail"}.issubset(neighbor["edge"].keys()))
                self.assertIn("node", neighbor)

            for target in ["create_user", node_id]:
                impact = run_lode_json(
                    data_dir,
                    "impact",
                    target,
                    "--repo",
                    str(repo),
                    "--direction",
                    "both",
                    "--depth",
                    "2",
                    "--max-nodes",
                    "100",
                    "--json",
                )
                self.assertTrue(impact["ok"])
                self.assertEqual(impact["query"], target)
                self.assertTrue(impact["results"])
                self.assertTrue(
                    {
                        "target",
                        "summary",
                        "files",
                        "callers",
                        "callees",
                        "blast_radius",
                        "incoming",
                        "outgoing",
                    }.issubset(impact["results"][0].keys())
                )
                for node in radius_nodes(impact):
                    start_line, end_line = node["lines"]
                    self.assertGreaterEqual(start_line, 1)
                    self.assertGreaterEqual(end_line, start_line)

            missing = run_command(
                [
                    sys.executable,
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "index",
                    str(repo / "does-not-exist"),
                    "--json",
                ],
                check=False,
            )
            self.assertNotEqual(missing.returncode, 0)
            missing_payload = json.loads(missing.stderr)
            self.assertFalse(missing_payload["ok"])
            self.assertTrue(missing_payload["error"])
            self.assertNotIn("Traceback", missing.stdout + missing.stderr)

            usage = run_command(
                [sys.executable, "-m", "lode", "index", "--json"],
                check=False,
            )
            self.assertNotEqual(usage.returncode, 0)
            self.assertNotIn("Traceback", usage.stdout + usage.stderr)

    def test_data_dir_isolation_in_cli_and_benchmark_flows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"
            home = root / "home"
            repo_a.mkdir()
            repo_b.mkdir()
            home.mkdir()
            write_fixture_repo(repo_a, "AlphaService")
            write_fixture_repo(repo_b, "BetaService")
            data_a = root / "data-a"
            data_b = root / "data-b"
            env = command_env({"HOME": str(home)})

            run_lode_json(data_a, "index", str(repo_a), "--json", env=env)
            env_b = command_env({"HOME": str(home), "LODE_DATA_DIR": str(data_b)})
            run_lode_json(None, "index", str(repo_b), "--json", env=env_b)

            status_a = run_lode_json(data_a, "status", "--json", env=env)
            status_b = run_lode_json(None, "status", "--json", env=env_b)
            self.assertEqual(status_a["repos"][0]["root"], str(repo_a.resolve()))
            self.assertEqual(status_b["repos"][0]["root"], str(repo_b.resolve()))

            alpha = run_lode_json(
                data_a,
                "search",
                "AlphaService",
                "--repo",
                str(repo_a),
                "--json",
                env=env,
            )
            beta_from_a = run_lode_json(
                data_a,
                "symbol",
                "BetaService",
                "--repo",
                str(repo_a),
                "--json",
                env=env,
            )
            beta = run_lode_json(
                None,
                "context",
                "BetaService save_user",
                "--repo",
                str(repo_b),
                "--limit",
                "5",
                "--json",
                env=env_b,
            )
            self.assertTrue(alpha["results"])
            self.assertFalse(beta_from_a["results"])
            self.assertTrue(beta["must_read"])

            bench = run_command(
                [
                    sys.executable,
                    "scripts/bench_lode.py",
                    "--repo",
                    str(repo_a),
                    "--data-dir",
                    str(data_a),
                    "--reset",
                    "--repeat",
                    "1",
                    "--query",
                    "AlphaService",
                    "--symbol",
                    "AlphaService",
                    "--json",
                ],
                env=env,
            )
            bench_payload = json.loads(bench.stdout)
            self.assertTrue(bench_payload["ok"])
            self.assertEqual(Path(bench_payload["data_dir"]).resolve(), data_a.resolve())
            self.assertFalse((home / ".local" / "share" / "lode" / "lode.sqlite").exists())


class PublicDaemonCompatibilityTests(unittest.TestCase):
    def test_daemon_api_contracts_errors_openapi_and_cli_consistency(self) -> None:
        assert_port_free()
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_fixture_repo(repo)
            cli_index = run_lode_json(data_dir, "index", str(repo), "--json")
            assert_index_stats(self, cli_index, repo)

            class Handler(LodeHandler):
                daemon_data_dir = data_dir
                daemon_metrics = Metrics()

            server = ReusableThreadingHTTPServer(("127.0.0.1", VALIDATION_PORT), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                status, headers, health = request_json(
                    "GET",
                    "/health",
                    headers={"x-request-id": "contract-test-health"},
                )
                self.assertEqual(status, 200)
                self.assertTrue(headers["content-type"].startswith("application/json"))
                self.assertEqual(headers["x-request-id"], "contract-test-health")
                self.assertEqual(health, {"ok": True, "service": "lode"})

                metrics_status, metrics_headers, metrics = request_text("/metrics")
                self.assertEqual(metrics_status, 200)
                self.assertTrue(metrics_headers["content-type"].startswith("text/plain"))
                self.assertIn("loded_requests_total", metrics)

                _, _, api_index = request_json("POST", "/index", payload={"path": str(repo)})
                assert_index_stats(self, api_index, repo)

                _, _, api_status = request_json("GET", "/status")
                self.assertTrue(api_status["ok"])
                for field in ["id", "root", "name", "indexed_at", "files", "nodes"]:
                    self.assertIn(field, api_status["repos"][0])

                query = urllib.parse.urlencode(
                    {"q": "UserService save_user", "repo": str(repo), "limit": 5}
                )
                _, _, api_search_q = request_json("GET", f"/search?{query}")
                query_alias = urllib.parse.urlencode(
                    {"query": "UserService save_user", "repo": str(repo), "limit": 5}
                )
                _, _, api_search_query = request_json("GET", f"/search?{query_alias}")
                _, _, api_search_post = request_json(
                    "POST",
                    "/search",
                    payload={"query": "UserService save_user", "repo": str(repo), "limit": 5},
                )
                for payload in [api_search_q, api_search_query, api_search_post]:
                    self.assertTrue(payload["ok"])
                    self.assertLessEqual(len(payload["results"]), 5)
                    self.assertTrue(payload["results"])
                    for result in payload["results"]:
                        assert_node_schema(self, result)
                        self.assertEqual(result["repo_id"], cli_index["repo_id"])

                _, _, api_context = request_json(
                    "POST",
                    "/context",
                    payload={
                        "query": "UserService save_user",
                        "repo": str(repo),
                        "budget": 2000,
                        "limit": 5,
                    },
                )
                self.assertTrue(api_context["ok"])
                self.assertEqual(api_context["query"], "UserService save_user")
                self.assertEqual(api_context["budget"], 2000)
                self.assertLessEqual(len(api_context["top_hits"]), 5)
                self.assertTrue(api_context["must_read"])

                get_impact_query = urllib.parse.urlencode(
                    {
                        "target": "create_user",
                        "repo": str(repo),
                        "direction": "up",
                        "depth": 1,
                        "max_nodes": 100,
                    }
                )
                _, _, api_impact_get = request_json("GET", f"/impact?{get_impact_query}")
                self.assertTrue(api_impact_get["ok"])
                self.assertEqual(first_impact_summary(api_impact_get)["direction"], "up")
                self.assertFalse(api_impact_get["results"][0]["blast_radius"]["downstream"])

                _, _, api_impact_post = request_json(
                    "POST",
                    "/impact",
                    payload={
                        "target": "create_user",
                        "repo": str(repo),
                        "direction": "both",
                        "depth": 2,
                        "max_nodes": 100,
                    },
                )
                self.assertTrue(api_impact_post["results"])
                for field in [
                    "target",
                    "summary",
                    "files",
                    "callers",
                    "callees",
                    "blast_radius",
                    "incoming",
                    "outgoing",
                ]:
                    self.assertIn(field, api_impact_post["results"][0])

                cli_search = run_lode_json(
                    data_dir,
                    "search",
                    "UserService save_user",
                    "--repo",
                    str(repo),
                    "--limit",
                    "5",
                    "--json",
                )
                cli_context = run_lode_json(
                    data_dir,
                    "context",
                    "UserService save_user",
                    "--repo",
                    str(repo),
                    "--budget",
                    "2000",
                    "--limit",
                    "5",
                    "--json",
                )
                cli_impact = run_lode_json(
                    data_dir,
                    "impact",
                    "create_user",
                    "--repo",
                    str(repo),
                    "--depth",
                    "2",
                    "--max-nodes",
                    "100",
                    "--json",
                )
                self.assertEqual(
                    normalized_search_tuples(cli_search), normalized_search_tuples(api_search_q)
                )
                self.assertEqual(
                    context_paths(cli_context, "top_hits"), context_paths(api_context, "top_hits")
                )
                self.assertEqual(
                    context_paths(cli_context, "must_read"), context_paths(api_context, "must_read")
                )
                for key in ["callers", "callees", "files", "upstream", "downstream"]:
                    self.assertEqual(
                        first_impact_summary(cli_impact)[key],
                        first_impact_summary(api_impact_post)[key],
                    )

                for method, path in [("GET", "/missing"), ("POST", "/missing")]:
                    _, headers, payload = request_json(
                        method,
                        path,
                        payload={} if method == "POST" else None,
                        expected_status=404,
                    )
                    self.assertTrue(headers["content-type"].startswith("application/json"))
                    self.assertEqual(payload, {"ok": False, "error": "not found"})

                invalid_post_cases = [
                    ("POST", "/search", None, b"{not-json"),
                    ("POST", "/search", None, b"[]"),
                    ("POST", "/context", {}, None),
                    ("POST", "/index", {"path": str(repo / "missing")}, None),
                ]
                for method, path, invalid_payload, raw in invalid_post_cases:
                    with self.subTest(path=path, payload=invalid_payload, raw=raw):
                        _, headers, error = request_json(
                            method,
                            path,
                            payload=invalid_payload,
                            raw=raw,
                            headers={"content-type": "application/json"} if raw else None,
                            expected_status=400,
                        )
                        self.assertTrue(headers["content-type"].startswith("application/json"))
                        self.assertFalse(error["ok"])
                        self.assertTrue(error["error"])

                invalid_get_paths = [
                    "/search?limit=not-int",
                    "/impact?target=create_user&depth=not-int",
                    "/impact?target=create_user&max_nodes=not-int",
                    "/impact?target=create_user&direction=sideways",
                    "/impact",
                ]
                for path in invalid_get_paths:
                    with self.subTest(path=path):
                        _, headers, error = request_json("GET", path, expected_status=400)
                        self.assertTrue(headers["content-type"].startswith("application/json"))
                        self.assertFalse(error["ok"])
                        self.assertTrue(error["error"])

                _, _, health_after_errors = request_json("GET", "/health")
                self.assertTrue(health_after_errors["ok"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        schema = (PROJECT_ROOT / "openapi" / "loded.openapi.yml").read_text(encoding="utf-8")
        for endpoint in [
            "/health:",
            "/metrics:",
            "/status:",
            "/search:",
            "/index:",
            "/context:",
            "/impact:",
        ]:
            self.assertIn(endpoint, schema)
        for param in [
            "name: q",
            "name: query",
            "name: repo",
            "name: limit",
            "name: target",
            "name: neighbor_limit",
            "name: depth",
            "name: max_nodes",
            "name: direction",
        ]:
            self.assertIn(param, schema)
        self.assertGreaterEqual(schema.count('"400"'), 5)
        self.assertGreaterEqual(schema.count('"404"'), 1)

    def test_lode_serve_mirrors_loded_for_core_api_shapes(self) -> None:
        assert_port_free()
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_fixture_repo(repo)
            run_lode_json(data_dir, "index", str(repo), "--json")
            common_args = ["--host", "127.0.0.1", "--port", str(VALIDATION_PORT)]
            daemon_commands = [
                [sys.executable, "-m", "lode.daemon", "--data-dir", str(data_dir), *common_args],
                [
                    sys.executable,
                    "-m",
                    "lode",
                    "--data-dir",
                    str(data_dir),
                    "serve",
                    *common_args,
                ],
            ]
            responses: list[dict[str, Any]] = []
            for command in daemon_commands:
                proc = subprocess.Popen(
                    command,
                    cwd=PROJECT_ROOT,
                    env=command_env(),
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                )
                try:
                    wait_for_health()
                    query = urllib.parse.urlencode(
                        {"q": "UserService", "repo": str(repo), "limit": 5}
                    )
                    _, _, health = request_json("GET", "/health")
                    _, _, status = request_json("GET", "/status")
                    _, _, search = request_json("GET", f"/search?{query}")
                    responses.append(
                        {
                            "health": health,
                            "status_keys": sorted(status.keys()),
                            "repo_roots": [entry["root"] for entry in status["repos"]],
                            "search_tuples": normalized_search_tuples(search),
                        }
                    )
                finally:
                    terminate_process(proc)
                    assert_port_released()

            self.assertEqual(responses[0], responses[1])


if __name__ == "__main__":
    unittest.main()
