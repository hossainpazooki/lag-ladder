"""Adapted from linear-ceiling tests/test_imports.py: three pins instead of one; this repo's placeholders."""
import re
import subprocess

import pytest

import lag_ladder


def test_repo_root_is_the_repo():
    assert (lag_ladder.REPO_ROOT / "pyproject.toml").exists()


def test_every_pin_in_upstream_md_is_declared_and_vice_versa():
    text = (lag_ladder.REPO_ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
    shas = set(re.findall(r"\b[0-9a-f]{40}\b", text))
    assert shas == {lag_ladder.INSTRUMENT_SHA, lag_ladder.CHASSIS_SHA, lag_ladder.PRODUCER_OBSERVED_SHA}, shas


def test_no_source_file_imports_the_instrument():
    for p in (lag_ladder.REPO_ROOT / "src").rglob("*.py"):
        assert not re.search(r"^\s*(from|import)\s+kvt\b", p.read_text(encoding="utf-8"), re.M), p.name


def test_verbatim_docs_present():
    assert (lag_ladder.REPO_ROOT / "docs" / "2026-09-19-holdover-design.md").exists()


_ALLOWED_TRACKED_RESULTS_PATHS = {"results/.gitkeep", "results/fstar/.gitkeep"}


def _tracked_results_paths() -> list[str]:
    try:
        proc = subprocess.run(["git", "ls-files", "--", "results"], cwd=lag_ladder.REPO_ROOT,
                              capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.fail(f"could not query git for tracked results/ paths: {exc}")
    return [line for line in proc.stdout.splitlines() if line]


def _assert_no_result_artifacts_tracked(tracked_paths) -> None:
    extras = sorted(set(tracked_paths) - _ALLOWED_TRACKED_RESULTS_PATHS)
    assert not extras, f"unexpected tracked files under results/: {extras}"


def test_results_tree_is_empty_placeholder():
    _assert_no_result_artifacts_tracked(_tracked_results_paths())


def test_results_tree_check_catches_tracked_artifact():
    with pytest.raises(AssertionError):
        _assert_no_result_artifacts_tracked(sorted(_ALLOWED_TRACKED_RESULTS_PATHS | {"results/fstar/x/report.json"}))
