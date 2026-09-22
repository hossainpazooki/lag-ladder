"""The one pin check: a missing upstream checkout is a refusal with the `{who} REFUSED` prefix, not a
NotADirectoryError from git (the 2026-09-06 workspace rename crashed `e8 --check` this way)."""
import pytest

from lag_ladder.upstream_gate import check_upstream


def test_missing_upstream_dir_refuses_by_name(tmp_path):
    with pytest.raises(RuntimeError, match=r"E8 REFUSED: upstream checkout .* does not exist"):
        check_upstream(tmp_path / "no-such-checkout", "a" * 40, ("kvt",), who="E8")


def test_placeholder_sha_still_refuses_before_touching_the_path(tmp_path):
    with pytest.raises(RuntimeError, match="not a commit sha"):
        check_upstream(tmp_path / "no-such-checkout", "UPSTREAM_SHA_PENDING", ("kvt",), who="E9")
