from __future__ import annotations

import argparse
import sys
from pathlib import Path

ENDPOINTS = [
    ("GET", "/health", "Daemon health check."),
    ("GET", "/metrics", "Prometheus text metrics for the local daemon."),
    ("GET", "/status", "Indexed repository status."),
    ("GET", "/search", "Search indexed code and docs."),
    ("GET", "/impact", "Impact report for a symbol or node."),
    ("POST", "/index", "Index a repository path."),
    ("POST", "/search", "Search indexed code and docs."),
    ("POST", "/context", "Build a bounded agent context pack."),
    ("POST", "/impact", "Impact report for a symbol or node."),
]


def render() -> str:
    rows = "\n".join(
        f"| `{method}` | `{path}` | {description} |" for method, path, description in ENDPOINTS
    )
    return (
        "# loded API\n\n"
        "Generated from `scripts/generate_openapi_docs.py`.\n\n"
        "| Method | Path | Description |\n"
        "|---|---|---|\n"
        f"{rows}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default="docs/api/loded.md")
    args = parser.parse_args(argv)
    output = Path(args.output)
    expected = render()
    if args.check:
        actual = output.read_text(encoding="utf-8") if output.exists() else ""
        if actual != expected:
            print(f"{output} is stale; run scripts/generate_openapi_docs.py", file=sys.stderr)
            return 1
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
