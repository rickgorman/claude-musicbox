#!/usr/bin/env python3
"""Pluggable local text embedder. Stdlib only (urllib) so it loads instantly.

Backends (MUSICBOX_EMBEDDER):
  ollama    (default) POST localhost:11434/api/embeddings, model all-minilm (384d).
            Ollama stays warm via keep_alive, so calls are ~20ms.
  http      POST any local embedding HTTP service that accepts
            {"text": ..., "query": true} and returns {"embeddings": [[...]]}.
            Configure with MUSICBOX_HTTP_URL.
  hashing   dependency-free MD5-bucket bag-of-words vector.
            Always available; weakest semantics.

Any failure (daemon down, timeout) returns None so the caller can fall back to
the lexicon path. Never raises into the hot path.
"""

import json
import os
import urllib.request

EMBEDDER = os.environ.get("MUSICBOX_EMBEDDER", "ollama")
OLLAMA_URL = os.environ.get("MUSICBOX_OLLAMA_URL", "http://localhost:11434/api/embeddings")
OLLAMA_MODEL = os.environ.get("MUSICBOX_OLLAMA_MODEL", "all-minilm")
HTTP_URL = os.environ.get("MUSICBOX_HTTP_URL", "http://localhost:3000/memory/api/v1/embed")
TIMEOUT = float(os.environ.get("MUSICBOX_EMBED_TIMEOUT", "0.6"))
HASH_DIM = 384


def backend_name():
    return EMBEDDER


def embed(text, backend=None):
    backend = backend or EMBEDDER
    try:
        if backend == "ollama":
            return _ollama(text)
        if backend == "http":
            return _http(text)
        if backend == "hashing":
            return _hashing(text)
    except Exception:
        return None
    return None


def _post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as response:
        return json.load(response)


def _ollama(text):
    body = _post(OLLAMA_URL, {"model": OLLAMA_MODEL, "prompt": text, "keep_alive": "30m"})
    return body.get("embedding")


def _http(text):
    body = _post(HTTP_URL, {"text": text, "query": True})
    vectors = body.get("embeddings") or []
    return vectors[0] if vectors else None


def _hashing(text):
    import hashlib

    vector = [0.0] * HASH_DIM
    for token in _tokens(text):
        bucket = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % HASH_DIM
        vector[bucket] += 1.0
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def _tokens(text):
    import re

    return re.findall(r"\w+", text.lower())
