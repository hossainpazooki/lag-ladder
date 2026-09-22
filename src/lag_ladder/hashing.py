"""Canonical bytes and SHA-256 for sealed artifacts.

A seal hashes the *parsed, canonicalised* JSON (sorted keys, 2-space indent, LF, UTF-8,
trailing newline), never the on-disk bytes. Reason: a Windows CRLF checkout must not break
a seal made on Linux. Lesson carried from passed-vs-true-demo's `canonicalBytes`
content-hash pinning, where hashing post-normalisation bytes was what made the pin
portable.
"""
import hashlib
import json
from pathlib import Path


def canonical_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_json_obj(obj) -> str:
    return sha256_hex(canonical_bytes(obj))


def hash_json_file(path) -> str:
    return hash_json_obj(json.loads(Path(path).read_text(encoding="utf-8")))


def sha256_file_bytes(path) -> str:
    """Raw-bytes digest, for binary or downloaded artifacts (trace files, dumps, token files)."""
    return sha256_hex(Path(path).read_bytes())


def sha256_text_file(path) -> str:
    """Newline-normalized UTF-8 digest for TEXT artifacts recorded in the ledger (TOML configs).

    A CRLF checkout (git autocrlf on Windows) must produce the same `config_sha256` as the LF
    file the ledger entries cite (0013-0022 cite 6915666d452d for config/e7.toml); for an LF
    file this equals `sha256_file_bytes`, so recorded values stay valid."""
    text = Path(path).read_text(encoding="utf-8")          # universal newlines -> "\n"
    return sha256_hex(text.encode("utf-8"))
