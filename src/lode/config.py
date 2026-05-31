from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "lode"


def default_data_dir() -> Path:
    override = os.environ.get("LODE_DATA_DIR") or os.environ.get("KG_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local" / "share" / APP_NAME


def sqlite_path(data_dir: Path | None = None) -> Path:
    root = data_dir or default_data_dir()
    return root / "lode.sqlite"


def kuzu_path(data_dir: Path | None = None) -> Path:
    root = data_dir or default_data_dir()
    return root / "lode.kuzu"


def repo_id_for_root(root: Path) -> str:
    import hashlib

    resolved = str(root.expanduser().resolve())
    return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]

