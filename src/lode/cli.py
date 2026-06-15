from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.error
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import default_data_dir, kuzu_path, sqlite_path
from .context import build_context_pack
from .graph import impact_report, impact_targets
from .indexer import index_repo
from .storage import (
    connect,
    embedding_counts,
    find_symbol,
    get_neighbors,
    list_repos,
    pending_embedding_nodes,
    repo_filter,
    search_nodes,
    upsert_embedding,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        FileNotFoundError,
        json.JSONDecodeError,
        RuntimeError,
        sqlite3.Error,
        urllib.error.URLError,
        ValueError,
    ) as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lode", description="Fully local repository knowledge graph for agents."
    )
    parser.add_argument(
        "--data-dir", type=Path, default=default_data_dir(), help="Lode data directory"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="Index a repository")
    p.add_argument("path", type=Path)
    p.add_argument("--sync-kuzu", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("status", help="Show indexed repositories")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("search", help="Search indexed symbols, docs, and files")
    p.add_argument("query")
    p.add_argument("--repo")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("symbol", help="Find a symbol by exact or partial name")
    p.add_argument("name")
    p.add_argument("--repo")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_symbol)

    p = sub.add_parser("context", help="Build a token-budgeted context pack for an agent task")
    p.add_argument("query")
    p.add_argument("--repo")
    p.add_argument("--budget", type=int, default=6000)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_context)

    p = sub.add_parser("neighbors", help="Show direct graph neighbors for a node ID")
    p.add_argument("node_id")
    p.add_argument("--limit", type=int, default=80)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_neighbors)

    p = sub.add_parser(
        "impact", help="Show callers, callees, and blast radius around a symbol or node"
    )
    p.add_argument("target")
    p.add_argument("--repo")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--neighbor-limit", type=int, default=200)
    p.add_argument(
        "--depth",
        type=int,
        help="Optional BFS depth limit for noisy graphs. Omit, or pass 0, for full traversal",
    )
    p.add_argument(
        "--max-nodes",
        type=int,
        default=1000,
        help="Safety cap for nodes in blast radius",
    )
    p.add_argument(
        "--direction",
        choices=["up", "down", "both"],
        default="both",
        help="Traversal direction: up (dependents), down (dependencies), or both",
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_impact)

    p = sub.add_parser("kuzu-sync", help="Project SQLite facts into embedded Kuzu")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_kuzu_sync)

    p = sub.add_parser("embed", help="Embed queued nodes with the local embeddings service")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--url")
    p.add_argument("--model")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("serve", help="Run the local HTTP daemon")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7979)
    p.set_defaults(func=cmd_serve)
    return parser


def cmd_index(args: argparse.Namespace) -> int:
    stats = index_repo(args.path, sqlite_path(args.data_dir))
    output: dict[str, Any] = {"ok": True, **asdict(stats)}
    if args.sync_kuzu:
        kuzu = sync_kuzu(args)
        output["kuzu"] = kuzu
        output["nodes"] = kuzu["nodes"]
        output["edges"] = kuzu["edges"]
    emit(output, args.json)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    with closing(connect(sqlite_path(args.data_dir))) as conn:
        output = {"ok": True, "data_dir": str(args.data_dir), "repos": list_repos(conn)}
    emit(output, args.json)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    with closing(connect(sqlite_path(args.data_dir))) as conn:
        repo_id = repo_filter(conn, args.repo)
        output = {
            "ok": True,
            "results": search_nodes(conn, args.query, repo_id=repo_id, limit=args.limit),
        }
    emit(output, args.json)
    return 0


def cmd_symbol(args: argparse.Namespace) -> int:
    with closing(connect(sqlite_path(args.data_dir))) as conn:
        repo_id = repo_filter(conn, args.repo)
        output = {
            "ok": True,
            "results": find_symbol(conn, args.name, repo_id=repo_id, limit=args.limit),
        }
    emit(output, args.json)
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    with closing(connect(sqlite_path(args.data_dir))) as conn:
        output = {
            "ok": True,
            **build_context_pack(
                conn,
                args.query,
                repo_path=args.repo,
                budget=args.budget,
                limit=args.limit,
            ),
        }
    emit(output, args.json)
    return 0


def cmd_neighbors(args: argparse.Namespace) -> int:
    with closing(connect(sqlite_path(args.data_dir))) as conn:
        output = {"ok": True, **get_neighbors(conn, args.node_id, limit=args.limit)}
    emit(output, args.json)
    return 0


def cmd_impact(args: argparse.Namespace) -> int:
    with closing(connect(sqlite_path(args.data_dir))) as conn:
        repo_id = repo_filter(conn, args.repo)
        targets = impact_targets(conn, args.target, repo_id, limit=args.limit)
        output = {
            "ok": True,
            "query": args.target,
            "results": [
                impact_report(
                    conn,
                    target,
                    neighbor_limit=args.neighbor_limit,
                    depth=args.depth,
                    max_nodes=args.max_nodes,
                    direction=args.direction,
                )
                for target in targets
            ],
        }
    emit(output, args.json)
    return 0


def cmd_kuzu_sync(args: argparse.Namespace) -> int:
    output = {"ok": True, "kuzu": sync_kuzu(args)}
    emit(output, args.json)
    return 0


def sync_kuzu(args: argparse.Namespace) -> dict[str, Any]:
    from .kuzu_store import sync_from_sqlite

    with closing(connect(sqlite_path(args.data_dir))) as conn:
        return sync_from_sqlite(conn, kuzu_path(args.data_dir))


def cmd_embed(args: argparse.Namespace) -> int:
    from .embeddings import embed_texts, embeddings_model

    with closing(connect(sqlite_path(args.data_dir))) as conn:
        nodes = pending_embedding_nodes(conn, limit=args.limit)
        if not nodes:
            output = {"ok": True, "embedded": 0, **embedding_counts(conn)}
            emit(output, args.json)
            return 0

        texts = [embedding_text(node) for node in nodes]
        vectors = embed_texts(texts, url=args.url)
        if len(vectors) != len(nodes):
            raise RuntimeError(
                f"Embedding endpoint returned {len(vectors)} vectors for {len(nodes)} texts"
            )
        model = args.model or embeddings_model()
        for node, vector in zip(nodes, vectors):
            upsert_embedding(conn, node["id"], node["repo_id"], vector, model)
        output = {"ok": True, "embedded": len(vectors), **embedding_counts(conn)}
    emit(output, args.json)
    return 0


def embedding_text(node: dict[str, Any]) -> str:
    parts = [
        str(node.get("qname") or node.get("name") or ""),
        str(node.get("signature") or ""),
        str(node.get("doc") or ""),
    ]
    return "\n".join(part for part in parts if part)


def cmd_serve(args: argparse.Namespace) -> int:
    from .daemon import serve

    serve(host=args.host, port=args.port, data_dir=args.data_dir)
    return 0


def emit(data: dict[str, Any], as_json: bool) -> None:
    if as_json or not sys.stdout.isatty():
        print(json.dumps(data, indent=2, sort_keys=True))
        return
    if "results" in data and data.get("query") is None:
        for row in data["results"]:
            print(f"{row['kind']:12} {row['qname']} {row['path']}:{row['start_line']}")
    elif "repos" in data:
        for repo in data["repos"]:
            print(
                f"{repo['name']:24} files={repo['files']} nodes={repo['nodes']} root={repo['root']}"
            )
    elif "results" in data and data.get("query") is not None:
        for result in data["results"]:
            target = result.get("target") or {}
            summary = result.get("summary") or {}
            qname = target.get("qname") or target.get("name") or data["query"]
            kind = target.get("kind") or "?"
            print(f"  {kind} {qname}")
            print(
                f"    callers={summary.get('callers', 0)}  callees={summary.get('callees', 0)}  "
                f"files={summary.get('files', 0)}"
            )
            print(
                f"    upstream={summary.get('upstream', 0)}  downstream={summary.get('downstream', 0)}  "
                f"entrypoints={summary.get('entrypoints', 0)}  depth={summary.get('depth', '?')}"
            )
            radius = result.get("blast_radius") or {}
            for entry in radius.get("upstream", []):
                node = entry.get("node") or {}
                d = entry.get("distance", "?")
                via = entry.get("via", "?")
                conf = entry.get("confidence", "?")
                print(
                    f"    [up d={d} {via} {conf}] {node.get('kind', '?')} {node.get('qname', '?')}"
                )
            for entry in radius.get("downstream", []):
                node = entry.get("node") or {}
                d = entry.get("distance", "?")
                via = entry.get("via", "?")
                conf = entry.get("confidence", "?")
                print(
                    f"    [dn d={d} {via} {conf}] {node.get('kind', '?')} {node.get('qname', '?')}"
                )
            for fentry in radius.get("files", []):
                print(f"    [file d={fentry.get('distance', '?')}] {fentry.get('path', '?')}")
            for ep in radius.get("entrypoints", []):
                node = ep.get("node") or {}
                print(f"    [entrypoint] {node.get('kind', '?')} {node.get('qname', '?')}")
            if summary.get("truncated"):
                print("    (truncated - increase --max-nodes for more)")
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
