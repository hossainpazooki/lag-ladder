import json

from lag_ladder.hashing import (
    canonical_bytes, hash_json_file, hash_json_obj, sha256_file_bytes, sha256_text_file,
)


def test_text_file_sha_is_newline_normalized_and_equals_raw_for_lf(tmp_path):
    lf, crlf = tmp_path / "lf.toml", tmp_path / "crlf.toml"
    lf.write_bytes(b"[e7]\nread_mult = 0.1\n")
    crlf.write_bytes(b"[e7]\r\nread_mult = 0.1\r\n")
    assert sha256_text_file(lf) == sha256_text_file(crlf) == sha256_file_bytes(lf)
    assert sha256_file_bytes(crlf) != sha256_file_bytes(lf)          # the raw digest is what differed
    lf.write_bytes(b"[e7]\nread_mult = 0.2\n")
    assert sha256_text_file(lf) != sha256_text_file(crlf)


def test_canonical_bytes_sorts_keys_and_ends_with_lf():
    b = canonical_bytes({"b": 1, "a": [1, 2]})
    assert b == b'{\n  "a": [\n    1,\n    2\n  ],\n  "b": 1\n}\n'


def test_hash_is_insensitive_to_key_order_and_whitespace(tmp_path):
    p1 = tmp_path / "a.json"; p2 = tmp_path / "b.json"
    p1.write_text('{"x": 1, "y": {"k": [1,2]}}', encoding="utf-8")
    p2.write_bytes(b'{\r\n "y": {"k": [1, 2]},\r\n "x": 1\r\n}\r\n')   # CRLF, reordered
    assert hash_json_file(p1) == hash_json_file(p2) == hash_json_obj({"x": 1, "y": {"k": [1, 2]}})


def test_hash_changes_when_a_value_changes():
    assert hash_json_obj({"x": 1}) != hash_json_obj({"x": 2})


def test_non_ascii_survives_roundtrip():
    obj = {"pair": "qwen3-0.6b-to-1.7b", "note": "R²"}
    assert json.loads(canonical_bytes(obj).decode("utf-8")) == obj
