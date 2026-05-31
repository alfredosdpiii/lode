from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import default_data_dir, sqlite_path
from .context import build_context_pack
from .indexer import index_repo
from .storage import connect, list_repos, repo_filter, search_nodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loded", description="Lode local HTTP daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7979)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.data_dir)
    return 0


def serve(
    host: str = "127.0.0.1", port: int = 7979, data_dir: Path | None = None
) -> None:
    root = data_dir or default_data_dir()

    class Handler(LodeHandler):
        daemon_data_dir = root

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"loded listening on http://{host}:{port} data_dir={root}")
    server.serve_forever()


class LodeHandler(BaseHTTPRequestHandler):
    daemon_data_dir: Path = default_data_dir()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_json({"ok": True, "service": "lode"})
            return
        if parsed.path == "/status":
            with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                self.send_json({"ok": True, "repos": list_repos(conn)})
            return
        if parsed.path == "/search":
            params = parse_qs(parsed.query)
            query = first(params, "q") or first(params, "query") or ""
            repo = first(params, "repo")
            limit = int(first(params, "limit") or 20)
            with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                repo_id = repo_filter(conn, repo)
                self.send_json(
                    {
                        "ok": True,
                        "results": search_nodes(
                            conn, query, repo_id=repo_id, limit=limit
                        ),
                    }
                )
            return
        self.send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:
        try:
            body = self.read_json()
            if self.path == "/index":
                path = Path(required(body, "path"))
                stats = index_repo(path, sqlite_path(self.daemon_data_dir))
                self.send_json({"ok": True, **asdict(stats)})
                return
            if self.path == "/search":
                query = str(required(body, "query"))
                repo = body.get("repo")
                limit = int(body.get("limit") or 20)
                with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                    repo_id = repo_filter(conn, repo)
                    self.send_json(
                        {
                            "ok": True,
                            "results": search_nodes(
                                conn, query, repo_id=repo_id, limit=limit
                            ),
                        }
                    )
                return
            if self.path == "/context":
                query = str(required(body, "query"))
                repo = body.get("repo")
                budget = int(body.get("budget") or 6000)
                limit = int(body.get("limit") or 10)
                with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                    self.send_json(
                        {
                            "ok": True,
                            **build_context_pack(
                                conn, query, repo_path=repo, budget=budget, limit=limit
                            ),
                        }
                    )
                return
            self.send_json({"ok": False, "error": "not found"}, status=404)
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ) as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)

    def read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(size) if size else b"{}"
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise TypeError("request body must be a JSON object")
        return data

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        return


def first(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def required(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field: {key}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
