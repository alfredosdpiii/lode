from __future__ import annotations

import os

FEATURE_DEFAULTS = {
    "metrics": True,
}


def enabled(name: str) -> bool:
    if name not in FEATURE_DEFAULTS:
        raise KeyError(f"unknown feature flag: {name}")
    env_name = "LODE_FEATURE_" + name.upper().replace("-", "_")
    raw = os.environ.get(env_name)
    if raw is None:
        return FEATURE_DEFAULTS[name]
    return raw.strip().lower() in {"1", "true", "yes", "on"}
