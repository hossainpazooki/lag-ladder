"""Shared fixtures. Adapted from linear-ceiling tests/conftest.py: the git helpers and the seal fixtures
are verbatim; the safetensors snapshot builder is dropped (no weights are read in this repo)."""
import subprocess
from pathlib import Path

import pytest

from lag_ladder.config import ArtifactRoot, SealConfig

GIT_ID = ["-c", "user.name=test", "-c", "user.email=test@example.com"]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *GIT_ID, *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path) -> Path:
    """A fresh git repo with one commit, plus the artifact roots the seal config points at."""
    r = tmp_path / "repo"
    (r / "ledger" / "predictions").mkdir(parents=True)
    (r / "mappers").mkdir()
    (r / "results" / "mapper").mkdir(parents=True)
    up = tmp_path / "upstream"
    (up / "mappers").mkdir(parents=True)
    (up / "results" / "mapper").mkdir(parents=True)
    git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("x\n")
    git(r, "add", "README.md")
    git(r, "commit", "-q", "-m", "init")
    return r


@pytest.fixture
def seal_cfg(repo) -> SealConfig:
    up = repo.parent / "upstream"
    return SealConfig(
        predictions_dir=repo / "ledger" / "predictions",
        upstream_path=up,
        artifact_roots=(
            ArtifactRoot(repo / "mappers", "{pair}/**/k*.safetensors"),
            ArtifactRoot(repo / "results" / "mapper", "{pair}/**/r2.json"),
            ArtifactRoot(up / "mappers", "{pair}/**/k*.safetensors"),
            ArtifactRoot(up / "results" / "mapper", "{pair}/**/r2.json"),
        ),
    )


def commit_all(repo: Path, msg: str = "seal") -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
