"""The one upstream-pin check every experiment gate uses.

An experiment records the upstream sha it was registered against. Later registrations may
re-pin the upstream (new scripts land there), so requiring HEAD == pin would make an older
experiment's summarizer refuse the moment a NEWER experiment is registered -- a false alarm.
What actually matters is that the TOOLS the experiment invokes are the pinned bytes:

  - the pinned commit exists and is HEAD or an ancestor of HEAD;
  - the invoked paths are unchanged between the pin and HEAD;
  - the working tree is clean for those paths.

Anything else there (the operator's unrelated local edits, later scripts) is not this
experiment's business and is not checked here.
"""
import re
import subprocess
from pathlib import Path


def check_upstream(upstream: Path, pinned_sha: str, paths: tuple[str, ...], *, who: str) -> None:
    """Raise RuntimeError (prefixed `{who} REFUSED`) unless the pin holds for `paths`."""
    if not re.fullmatch(r"[0-9a-f]{40}", pinned_sha):
        raise RuntimeError(f"{who} REFUSED: upstream_sha {pinned_sha!r} is not a commit sha; the "
                           "registering entry's re-pin must be recorded before anything runs")
    upstream = Path(upstream)
    if not upstream.is_dir():   # a renamed/missing checkout is a refusal, not a NotADirectoryError from git (2026-09-06)
        raise RuntimeError(f"{who} REFUSED: upstream checkout {upstream} does not exist; UPSTREAM.md names the pinned "
                           "repo and its local path (../kv-transfer-replication)")
    anc = subprocess.run(["git", "merge-base", "--is-ancestor", pinned_sha, "HEAD"], cwd=upstream,
                         capture_output=True)
    if anc.returncode != 0:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=upstream, capture_output=True, text=True)
        raise RuntimeError(f"{who} REFUSED: pinned upstream commit {pinned_sha[:12]} is not an ancestor of "
                           f"HEAD {head.stdout.strip()[:12]} (missing, or history rewritten)")
    diff = subprocess.run(["git", "diff", "--quiet", pinned_sha, "HEAD", "--", *paths], cwd=upstream)
    if diff.returncode != 0:
        raise RuntimeError(f"{who} REFUSED: upstream paths {paths} changed between the pin "
                           f"{pinned_sha[:12]} and HEAD; the invoked tools are not the pinned bytes")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=upstream,
                           capture_output=True, text=True)
    if dirty.stdout.strip():
        raise RuntimeError(f"{who} REFUSED: upstream working tree is dirty under {paths}:\n{dirty.stdout}")
