from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LodeEndToEndTests(unittest.TestCase):
    def test_cli_daemon_and_kuzu_end_to_end(self) -> None:
        with (
            tempfile.TemporaryDirectory() as repo_tmp,
            tempfile.TemporaryDirectory() as data_tmp,
        ):
            repo = Path(repo_tmp)
            data_dir = Path(data_tmp)
            write_sample_repo(repo)

            index = run_lode(data_dir, "index", str(repo), "--json")
            self.assertGreater(index["nodes"], 0)
            self.assertGreater(index["edges"], 0)

            status = run_lode(data_dir, "status", "--json")
            self.assertEqual(status["repos"][0]["root"], str(repo.resolve()))

            search = run_lode(data_dir, "search", "create user", "--json")
            self.assertTrue(search["results"])

            symbol = run_lode(data_dir, "symbol", "create_user", "--json")
            self.assertEqual(symbol["results"][0]["kind"], "Function")
            node_id = symbol["results"][0]["id"]

            neighbors = run_lode(data_dir, "neighbors", node_id, "--json")
            self.assertEqual(neighbors["node"]["id"], node_id)

            context = run_lode(
                data_dir,
                "context",
                "where is the create user route handled",
                "--budget",
                "2000",
                "--json",
            )
            self.assertTrue(context["must_read"])

            kuzu = run_lode(data_dir, "kuzu-sync", "--json")
            self.assertGreater(kuzu["kuzu"]["nodes"], 0)
            self.assertGreater(kuzu["kuzu"]["edges"], 0)

            port = open_port()
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lode.daemon",
                    "--data-dir",
                    str(data_dir),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=PROJECT_ROOT,
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                base = f"http://127.0.0.1:{port}"
                wait_for_json(base + "/health")
                metrics = get_text(base + "/metrics")
                self.assertIn("loded_requests_total", metrics)
                api_index = post_json(base + "/index", {"path": str(repo)})
                self.assertTrue(api_index["ok"])
                api_status = get_json(base + "/status")
                self.assertTrue(api_status["repos"])
                query = urllib.parse.urlencode({"q": "create_user", "limit": 5})
                api_search = get_json(base + "/search?" + query)
                self.assertTrue(api_search["results"])
                api_context = post_json(
                    base + "/context",
                    {"query": "create user route", "budget": 2000, "limit": 5},
                )
                self.assertTrue(api_context["must_read"])
            finally:
                terminate_process(proc)

    @unittest.skipUnless(
        os.environ.get("RUN_LODE_DOCKER_E2E") == "1",
        "set RUN_LODE_DOCKER_E2E=1 to run Docker Compose e2e",
    )
    def test_docker_compose_stack_and_real_embeddings_end_to_end(self) -> None:
        if shutil.which("docker") is None:
            self.skipTest("docker is not installed")
        with tempfile.TemporaryDirectory() as tmp_home:
            temp_root = Path(tmp_home)
            data_dir = temp_root / "data"
            data_dir.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "LODE_DATA_DIR_HOST": str(data_dir),
                    "LODE_HF_CACHE_DIR": str(Path.home() / ".cache" / "lode" / "hf-arctic-s"),
                    "LODE_REPOS_DIR": str(PROJECT_ROOT.parent),
                    "LODE_UID": str(os.getuid()),
                    "LODE_GID": str(os.getgid()),
                }
            )
            try:
                subprocess.run(
                    ["docker", "compose", "up", "-d", "--build"],
                    check=True,
                    cwd=PROJECT_ROOT,
                    env=env,
                    timeout=180,
                )
                wait_for_json("http://127.0.0.1:7979/health", timeout=120)
                vectors = wait_for_embeddings("http://127.0.0.1:7980/embed")
                self.assertEqual(len(vectors), 2)
                self.assertEqual(len(vectors[0]), 384)

                api_index = post_json("http://127.0.0.1:7979/index", {"path": "/app"})
                self.assertTrue(api_index["ok"])
                api_search = get_json("http://127.0.0.1:7979/search?q=build_context_pack")
                self.assertTrue(api_search["results"])

                embed = run_lode(
                    data_dir,
                    "embed",
                    "--limit",
                    "5",
                    "--json",
                    env={
                        "LODE_EMBEDDINGS_URL": "http://127.0.0.1:7980",
                        "LODE_EMBEDDINGS_MODEL": "Snowflake/snowflake-arctic-embed-s",
                    },
                )
                self.assertEqual(embed["embedded"], 5)
                with closing(sqlite3.connect(data_dir / "lode.sqlite")) as conn:
                    count, min_dims, max_dims = conn.execute(
                        "SELECT COUNT(*), MIN(dims), MAX(dims) FROM embeddings"
                    ).fetchone()
                self.assertEqual(count, 5)
                self.assertEqual((min_dims, max_dims), (384, 384))
            finally:
                subprocess.run(
                    ["docker", "compose", "down", "--remove-orphans"],
                    cwd=PROJECT_ROOT,
                    env=env,
                    check=False,
                    timeout=60,
                )


def run_lode(data_dir: Path, *args: str, env: dict[str, str] | None = None) -> dict:
    command_env = os.environ.copy()
    if env:
        command_env.update(env)
    result = subprocess.run(
        [sys.executable, "-m", "lode", "--data-dir", str(data_dir), *args],
        check=True,
        capture_output=True,
        cwd=PROJECT_ROOT,
        env=command_env,
        text=True,
        timeout=120,
    )
    return json.loads(result.stdout)


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url: str, timeout: float = 10) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def post_json(url: str, payload: dict, timeout: float = 30) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_json(url: str, timeout: float = 30) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            return get_json(url, timeout=5)
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def wait_for_embeddings(url: str, timeout: float = 120) -> list[list[float]]:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            request = urllib.request.Request(
                url,
                data=json.dumps({"inputs": ["hello world", "repository graph"]}).encode("utf-8"),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8"))
        except (
            OSError,
            TimeoutError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            last_error = exc
            time.sleep(1)
    raise TimeoutError(f"Timed out waiting for embeddings: {last_error}")


def open_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def terminate_process(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)


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
