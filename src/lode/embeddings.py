from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from ipaddress import ip_address
from typing import Any

LOCAL_EMBEDDING_HOSTS = {"localhost", "embeddings"}


def embeddings_url() -> str | None:
    return os.environ.get("LODE_EMBEDDINGS_URL") or os.environ.get("KG_EMBEDDINGS_URL")


def embeddings_model() -> str:
    return (
        os.environ.get("LODE_EMBEDDINGS_MODEL")
        or os.environ.get("KG_EMBEDDINGS_MODEL")
        or "unknown"
    )


def local_embeddings_endpoint(url: str | None = None) -> str:
    endpoint = (url or embeddings_url() or "").rstrip("/")
    if not endpoint:
        raise RuntimeError("No local embeddings endpoint configured. Set LODE_EMBEDDINGS_URL.")
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("Embeddings endpoint must be a local HTTP URL.")
    host = parsed.hostname.lower()
    if host not in LOCAL_EMBEDDING_HOSTS:
        try:
            if not ip_address(host).is_loopback:
                raise ValueError
        except ValueError as exc:
            raise RuntimeError(
                "Embeddings endpoint must be local, use loopback or Docker service 'embeddings'."
            ) from exc
    return endpoint


def embed_texts(texts: list[str], url: str | None = None, retries: int = 2) -> list[list[float]]:
    endpoint = local_embeddings_endpoint(url)
    payload = json.dumps({"inputs": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{endpoint}/embed",
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    data: Any = None
    last_error: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt >= retries:
                raise RuntimeError("Embedding endpoint failed after retries") from exc
            time.sleep(0.25 * (2**attempt))
    else:
        raise RuntimeError("Embedding endpoint failed") from last_error
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("embeddings"), list):
        return data["embeddings"]
    raise RuntimeError("Unexpected embeddings response shape")
