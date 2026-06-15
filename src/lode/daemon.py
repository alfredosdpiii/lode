from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .config import default_data_dir, sqlite_path
from .context import build_context_pack
from .features import enabled
from .graph import impact_report, impact_targets
from .indexer import index_repo
from .observability import Metrics, log_event
from .storage import connect, list_repos, repo_filter, search_nodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loded", description="Lode local HTTP daemon")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7979)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = parser.parse_args(argv)
    serve(args.host, args.port, args.data_dir)
    return 0


def serve(host: str = "127.0.0.1", port: int = 7979, data_dir: Path | None = None) -> None:
    root = data_dir or default_data_dir()

    class Handler(LodeHandler):
        daemon_data_dir = root

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"loded listening on http://{host}:{port} data_dir={root}")
    server.serve_forever()


class LodeHandler(BaseHTTPRequestHandler):
    daemon_data_dir: Path = default_data_dir()
    daemon_metrics: Metrics = Metrics()

    def do_GET(self) -> None:
        self.ensure_request_id()
        try:
            self.handle_get()
        except (
            FileNotFoundError,
            json.JSONDecodeError,
            OSError,
            sqlite3.Error,
            TypeError,
            ValueError,
        ) as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=400)

    def handle_get(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.send_json({"ok": True, "service": "lode"})
            return
        if parsed.path == "/metrics" and enabled("metrics"):
            self.send_text(self.daemon_metrics.render(), content_type="text/plain")
            return
        if parsed.path == "/status":
            with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                self.send_json({"ok": True, "repos": list_repos(conn)})
            return
        if parsed.path == "/search":
            params = parse_qs(parsed.query)
            query = first(params, "q") or first(params, "query") or ""
            repo = first(params, "repo")
            limit = parse_query_int(params, "limit", 20)
            with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                repo_id = repo_filter(conn, repo)
                self.send_json(
                    {
                        "ok": True,
                        "results": search_nodes(conn, query, repo_id=repo_id, limit=limit),
                    }
                )
            return
        if parsed.path == "/impact":
            params = parse_qs(parsed.query)
            target = required_query(params, "target", "q")
            repo = first(params, "repo")
            limit = parse_query_int(params, "limit", 5)
            neighbor_limit = parse_query_int(params, "neighbor_limit", 200)
            depth = parse_optional_query_int(params, "depth")
            max_nodes = parse_query_int(params, "max_nodes", 1000)
            direction = parse_direction(first(params, "direction"))
            with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                repo_id = repo_filter(conn, repo)
                targets = impact_targets(conn, target, repo_id, limit=limit)
                self.send_json(
                    {
                        "ok": True,
                        "query": target,
                        "results": [
                            impact_report(
                                conn,
                                t,
                                neighbor_limit=neighbor_limit,
                                depth=depth,
                                max_nodes=max_nodes,
                                direction=direction,
                            )
                            for t in targets
                        ],
                    }
                )
            return
        self.send_json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self) -> None:
        self.ensure_request_id()
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
                limit = parse_body_int(body, "limit", 20)
                with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                    repo_id = repo_filter(conn, repo)
                    self.send_json(
                        {
                            "ok": True,
                            "results": search_nodes(conn, query, repo_id=repo_id, limit=limit),
                        }
                    )
                return
            if self.path == "/context":
                query = str(required(body, "query"))
                repo = body.get("repo")
                budget = parse_body_int(body, "budget", 6000)
                limit = parse_body_int(body, "limit", 10)
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
            if self.path == "/impact":
                target = str(required(body, "target"))
                repo = body.get("repo")
                limit = parse_body_int(body, "limit", 5)
                neighbor_limit = parse_body_int(body, "neighbor_limit", 200)
                depth = parse_optional_body_int(body, "depth")
                max_nodes = parse_body_int(body, "max_nodes", 1000)
                direction = parse_direction(body.get("direction"))
                with closing(connect(sqlite_path(self.daemon_data_dir))) as conn:
                    repo_id = repo_filter(conn, repo)
                    targets = impact_targets(conn, target, repo_id, limit=limit)
                    self.send_json(
                        {
                            "ok": True,
                            "query": target,
                            "results": [
                                impact_report(
                                    conn,
                                    t,
                                    neighbor_limit=neighbor_limit,
                                    depth=depth,
                                    max_nodes=max_nodes,
                                    direction=direction,
                                )
                                for t in targets
                            ],
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

    def ensure_request_id(self) -> str:
        request_id = getattr(self, "request_id", None)
        if request_id:
            return request_id
        header = self.headers.get("x-request-id") if self.headers else None
        self.request_id = header or uuid.uuid4().hex
        return self.request_id

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.send_header("x-request-id", self.ensure_request_id())
        self.end_headers()
        self.wfile.write(data)
        self.record_request(status)

    def send_text(self, payload: str, status: int = 200, content_type: str = "text/plain") -> None:
        data = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(data)))
        self.send_header("x-request-id", self.ensure_request_id())
        self.end_headers()
        self.wfile.write(data)
        self.record_request(status)

    def record_request(self, status: int) -> None:
        path = urlparse(self.path).path
        self.daemon_metrics.record(self.command, path, status)
        log_event(
            "http_request",
            request_id=self.ensure_request_id(),
            method=self.command,
            path=path,
            status=status,
        )

    def log_message(self, format: str, *args: Any) -> None:
        _ = (format, args)
        return


def first(params: dict[str, list[str]], key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def required_query(params: dict[str, list[str]], key: str, *aliases: str) -> str:
    for name in (key, *aliases):
        value = first(params, name)
        if value is not None and value != "":
            return value
    names = " or ".join((key, *aliases))
    raise ValueError(f"missing required query parameter: {names}")


def parse_int_value(value: Any, name: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def parse_query_int(params: dict[str, list[str]], key: str, default: int) -> int:
    value = first(params, key)
    if value in (None, ""):
        return default
    return parse_int_value(value, key)


def parse_optional_query_int(params: dict[str, list[str]], key: str) -> int | None:
    value = first(params, key)
    if value in (None, ""):
        return None
    return parse_int_value(value, key)


def parse_body_int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key)
    if value in (None, ""):
        return default
    return parse_int_value(value, key)


def parse_optional_body_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value in (None, ""):
        return None
    return parse_int_value(value, key)


def parse_direction(value: Any) -> str:
    direction = "both" if value in (None, "") else str(value)
    if direction not in {"up", "down", "both"}:
        raise ValueError("direction must be 'up', 'down', or 'both'")
    return direction


def required(data: dict[str, Any], key: str) -> Any:
    value = data.get(key)
    if value is None or value == "":
        raise ValueError(f"missing required field: {key}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
