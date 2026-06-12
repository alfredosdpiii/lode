from __future__ import annotations

import argparse
import cProfile
import pstats
import tempfile
from pathlib import Path

from lode.config import sqlite_path
from lode.indexer import index_repo


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile Lode indexing with cProfile.")
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--sort", default="cumtime")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    profiler = cProfile.Profile()
    with tempfile.TemporaryDirectory() as data_tmp:
        profiler.enable()
        index_repo(args.repo, sqlite_path(Path(data_tmp)))
        profiler.disable()
    stats = pstats.Stats(profiler).strip_dirs().sort_stats(args.sort)
    stats.print_stats(args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
