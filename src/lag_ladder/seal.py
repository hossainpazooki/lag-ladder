"""The seal (invariant 1): a screen prediction for an ordered pair is written, hashed and
committed BEFORE any mapper for that pair is fitted anywhere.

Two refusals make violation structurally awkward:
  * write_prediction refuses if a fitted-mapper artifact for the pair already exists in
    any configured results tree (local or the pinned upstream), or if any configured root
    is missing (fail closed), or if the pair is already sealed.
  * require_sealed (the E1+ runner gate) refuses unless the sealed file exists, its
    canonical hash matches the sidecar, it is committed, unchanged since HEAD, and never
    modified after the commit that sealed it.

Post-fit exception (decision D2, ledger entry 0002): the 0.6B->1.7B pair was fitted in the
upstream before this repo existed, so E1's prediction cannot be sealed pre-fit. Such a file
is written with post_fit=True, carries sealed_pre_fit=false plus the artifacts that
pre-dated it, and is only consumable by a runner that passes allow_post_fit=True (E1).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from lag_ladder import REPO_ROOT
from lag_ladder.config import SealConfig, load_seal_config
from lag_ladder.hashing import canonical_bytes, hash_json_obj


class SealViolation(RuntimeError):
    pass


def _validate_pair(pair: str) -> None:
    """Structural check only -- no ladder/model allowlist. Pairs added by later work must not
    require editing this validator, or it will be edited carelessly (or not at all). Rejects
    anything that could make `predictions_dir / f"{pair}.json"` resolve outside predictions_dir:
    path separators, '.'/'..' segments, absolute paths, and the empty string."""
    if not pair:
        raise SealViolation("pair name must not be empty")
    if "/" in pair or "\\" in pair:
        raise SealViolation(f"pair name {pair!r} contains a path separator; not a valid pair name")
    if pair in (".", ".."):
        raise SealViolation(f"pair name {pair!r} is a '.' or '..' segment; not a valid pair name")
    if Path(pair).is_absolute():
        raise SealViolation(f"pair name {pair!r} is an absolute path; not a valid pair name")


def prediction_path(pair: str, cfg: SealConfig) -> Path:
    _validate_pair(pair)
    return cfg.predictions_dir / f"{pair}.json"


def sidecar_path(pair: str, cfg: SealConfig) -> Path:
    _validate_pair(pair)
    return cfg.predictions_dir / f"{pair}.sha256"


def find_mapper_artifacts(pair: str, cfg: SealConfig) -> list[Path]:
    _validate_pair(pair)
    hits: list[Path] = []
    for root in cfg.artifact_roots:
        if not root.path.is_dir():
            raise SealViolation(
                f"artifact root {root.path} does not exist; refusing to seal {pair} because a "
                "fitted mapper could be hiding behind a missing tree (mount the upstream at the "
                "path in config/seal.toml, or fix the config)")
        hits.extend(sorted(p for p in root.path.glob(root.pattern.format(pair=pair)) if p.is_file()))
    return hits


def write_prediction(pair: str, payload: dict, cfg: SealConfig, *, post_fit: bool = False) -> str:
    hits = find_mapper_artifacts(pair, cfg)
    if hits and not post_fit:
        listing = "\n  ".join(h.as_posix() for h in hits)
        raise SealViolation(
            f"fitted mapper artifact(s) already exist for {pair}; a prediction written now would "
            f"not be pre-fit:\n  {listing}")
    if post_fit and not hits:
        raise SealViolation(
            f"post_fit=True but no fitted mapper exists for {pair}; a prediction that CAN be "
            "sealed pre-fit must be")
    p = prediction_path(pair, cfg)
    if p.exists() or sidecar_path(pair, cfg).exists():
        raise SealViolation(f"{pair} is already sealed at {p}; amendments are new ledger entries, not edits")
    record = {
        "pair": pair,
        "sealed_pre_fit": not post_fit,
        "prior_artifacts": [h.resolve().as_posix() for h in hits],
        "prediction": payload,
    }
    digest = hash_json_obj(record)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(canonical_bytes(record))
    sidecar_path(pair, cfg).write_text(digest + "\n", encoding="utf-8")
    return digest


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)


def _assert_committed_and_unmodified(path: Path, repo_root: Path) -> None:
    rel = path.resolve().relative_to(Path(repo_root).resolve()).as_posix()
    if _git(repo_root, "ls-files", "--error-unmatch", rel).returncode != 0:
        raise SealViolation(f"{rel} is not tracked by git; a seal that is not committed is not a seal")
    if _git(repo_root, "diff", "--quiet", "HEAD", "--", rel).returncode != 0:
        raise SealViolation(f"{rel} differs from HEAD (staged or unstaged); commit the seal first")
    # `git log --diff-filter=M` alone is not enough: git assigns no M status to a
    # delete-then-re-add (shows as D then A) or to a rename-away-then-rename-back (shows as
    # R at each end), so either bypasses a filter on M undetected. Plain `git log -- <path>`
    # (no --follow, so it does NOT chase renames past this exact literal path) lists every
    # commit that ever touched this exact pathname regardless of status letter -- add,
    # modify, delete, or either end of a rename. A genuine, untampered seal is touched by
    # exactly one commit ever: the one that sealed it. Any second commit touching the path
    # means it changed after sealing, whatever git calls the change.
    log = _git(repo_root, "log", "--format=%H", "--", rel)
    if log.returncode != 0:
        raise SealViolation(f"git log failed for {rel}: {log.stderr.strip()}")
    touched_in = log.stdout.split()
    if not touched_in:
        raise SealViolation(f"{rel} is tracked but has no commit history touching it (unexpected)")
    if len(touched_in) > 1:
        # git log lists newest first; the oldest entry is the commit that first introduced
        # the path (the sealing commit) and every newer one touching the path is offending.
        sealing_commit = touched_in[-1]
        offending = touched_in[:-1]
        raise SealViolation(
            f"{rel} was modified after sealing: sealed in commit {sealing_commit} but touched "
            f"again in {len(offending)} later commit(s) ({', '.join(offending)}) -- a sealed "
            "file must never be touched again after the commit that introduced it, whether by "
            "in-place edit, delete-then-re-add, or rename")


def require_sealed(pair: str, cfg: SealConfig, *, repo_root: Path, allow_post_fit: bool = False,
                   require_committed: bool = True) -> dict:
    p, s = prediction_path(pair, cfg), sidecar_path(pair, cfg)
    if not p.exists():
        raise SealViolation(f"no sealed prediction for {pair} at {p}; the screen must run and seal before any fit")
    if not s.exists():
        raise SealViolation(f"sealed prediction {p} has no sidecar hash {s}")
    record = json.loads(p.read_text(encoding="utf-8"))
    digest, expected = hash_json_obj(record), s.read_text(encoding="utf-8").strip()
    if digest != expected:
        raise SealViolation(f"hash mismatch for {pair}: file hashes to {digest}, sidecar says {expected}")
    if record.get("pair") != pair:
        raise SealViolation(f"{p} is sealed for pair {record.get('pair')!r}, not {pair!r}")
    if not record.get("sealed_pre_fit", False) and not allow_post_fit:
        raise SealViolation(
            f"{pair} carries a post-fit prediction (fitted before it was written); only the E1 "
            "identity check may consume it, and it never counts as a pre-fit claim")
    if require_committed:
        _assert_committed_and_unmodified(p, repo_root)
        _assert_committed_and_unmodified(s, repo_root)
    return record


def verify_all(cfg: SealConfig, repo_root: Path) -> list[str]:
    """CI entry. Hashes + immutability only; artifact roots are not consulted (CI has no upstream)."""
    d = cfg.predictions_dir
    jsons = sorted(d.glob("*.json")) if d.is_dir() else []
    sidecars = sorted(d.glob("*.sha256")) if d.is_dir() else []
    orphans = [s.name for s in sidecars if not (d / f"{s.stem}.json").exists()]
    if orphans:
        raise SealViolation(f"orphan sidecar(s) without a prediction file: {orphans}")
    if not jsons:
        return ["OK (no sealed predictions yet)"]
    out = []
    for p in jsons:
        rec = require_sealed(p.stem, cfg, repo_root=repo_root, allow_post_fit=True)
        kind = "pre-fit" if rec["sealed_pre_fit"] else "POST-FIT"
        out.append(f"OK {p.stem} {sidecar_path(p.stem, cfg).read_text().strip()} {kind}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lag_ladder.seal")
    ap.add_argument("--config", default=str(REPO_ROOT / "config" / "seal.toml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify")
    sp = sub.add_parser("show"); sp.add_argument("pair")
    wp = sub.add_parser("write"); wp.add_argument("--pair", required=True)
    wp.add_argument("--payload", required=True, help="JSON file produced by the screen")
    wp.add_argument("--post-fit", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_seal_config(Path(a.config), REPO_ROOT)
    try:
        if a.cmd == "verify":
            for line in verify_all(cfg, REPO_ROOT):
                print(line)
        elif a.cmd == "show":
            rec = require_sealed(a.pair, cfg, repo_root=REPO_ROOT, allow_post_fit=True, require_committed=False)
            print(json.dumps(rec, indent=2, sort_keys=True))
        else:
            payload = json.loads(Path(a.payload).read_text(encoding="utf-8"))
            print("sealed", a.pair, write_prediction(a.pair, payload, cfg, post_fit=a.post_fit))
    except SealViolation as e:
        print(f"SEAL VIOLATION: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
