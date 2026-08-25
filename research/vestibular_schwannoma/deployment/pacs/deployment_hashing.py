"""Dependency-free serialization and hashing helpers for deployment manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def bundle_sha256(schema_version: int, model_type: str, member_sha256s) -> str:
    """Hash the inference-relevant identity of an ordered model bundle."""
    payload = {
        "schema_version": schema_version,
        "model_type": model_type,
        "member_sha256s": list(member_sha256s),
    }
    return sha256_bytes(canonical_json(payload).encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
