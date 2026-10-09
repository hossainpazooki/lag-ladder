"""TOML config -> frozen dataclasses. Seeds and thresholds live here, never in code.

Adapted from linear-ceiling `src/linear_ceiling/config.py` (UPSTREAM.md provenance): the seal loader is
verbatim; the E0/E7/E8/E9 loaders are dropped; `load_pilot_config` is new.
"""
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath


@dataclass(frozen=True)
class ArtifactRoot:
    path: Path
    pattern: str


@dataclass(frozen=True)
class SealConfig:
    predictions_dir: Path
    upstream_path: Path
    artifact_roots: tuple[ArtifactRoot, ...]


def _read(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _resolve(repo_root: Path, raw: str, upstream: Path | None = None) -> Path:
    if "${upstream}" in raw:
        if upstream is None:
            raise ValueError("${upstream} used before upstream_path was known")
        raw = raw.replace("${upstream}", upstream.as_posix())
        return Path(raw).resolve()
    return (repo_root / raw)


def load_seal_config(path: Path, repo_root: Path) -> SealConfig:
    d = _read(Path(path))
    repo_root = Path(repo_root)
    upstream = repo_root / d["upstream_path"]
    roots = []
    for r in d.get("artifact_roots", []):
        if "{pair}" not in r["pattern"]:
            raise ValueError(f"artifact root pattern {r['pattern']!r} has no {{pair}} placeholder; "
                             "a root that cannot be scoped to a pair would match everything or nothing")
        roots.append(ArtifactRoot(_resolve(repo_root, r["path"], upstream), r["pattern"]))
    if not roots:
        raise ValueError("seal config lists no artifact_roots; the writer would have nothing to refuse on")
    return SealConfig(predictions_dir=repo_root / d["predictions_dir"],
                      upstream_path=upstream, artifact_roots=tuple(roots))


@dataclass(frozen=True)
class PilotConfig:
    """The (A)-only pilot on the OLMo revisions (design §9 step 3). `registered_by` is the four-digit
    ledger entry that fixed these values; empty means UNREGISTERED and a gate must refuse to run."""
    source: str
    base: str
    revisions: tuple[str, ...]
    stride_steps: int
    anchors: tuple[str, ...]
    results_dir: Path
    upstream_path: Path
    upstream_sha: str
    pair: str             # the upstream Pair name both checkpoints share (kvt/pairs.py)
    tokens: Path          # held-out token file, resolved under upstream_path
    tokens_sha256: str    # sha256 of that file, or a *_PENDING placeholder until registered
    heldout: tuple[int, int]   # [lo, hi) sequence indices of the token file read as held out
    dtype: str
    seed: int
    rule: dict            # statistic, tau_K, tau_ladder, holds_max, degrades_min
    controls: dict        # identity_max, scrambled_min
    registered_by: str
    config_path: Path


_RULE_KEYS = ("statistic", "tau_K", "tau_ladder", "holds_max", "degrades_min")
_CONTROL_KEYS = ("identity_max", "scrambled_min")
_DTYPES = ("float32", "bfloat16", "float16")   # the pinned instrument's --dtype choices


def load_pilot_config(path: Path, repo_root: Path) -> PilotConfig:
    path = Path(path)
    c = _read(path)["pilot"]
    for key in ("source", "base", "revisions", "stride_steps", "anchors", "results_dir",
                "upstream_path", "upstream_sha", "pair", "tokens", "tokens_sha256", "heldout", "dtype",
                "seed", "rule", "controls"):
        if key not in c:
            raise ValueError(f"{path.name} [pilot] is missing {key}")
    missing = [k for k in _RULE_KEYS if k not in c["rule"]]
    if missing:
        raise ValueError(f"{path.name} [pilot.rule] is missing {missing}")
    missing = [k for k in _CONTROL_KEYS if k not in c["controls"]]
    if missing:
        raise ValueError(f"{path.name} [pilot.controls] is missing {missing}")
    revs = tuple(str(r) for r in c["revisions"])
    if len(revs) < 2 or len(set(revs)) != len(revs):
        raise ValueError(f"{path.name} [pilot] revisions must be >= 2 distinct revision names")
    anchors = tuple(str(a) for a in c["anchors"])
    if not anchors or any(a not in revs for a in anchors):
        raise ValueError(f"{path.name} [pilot] every anchor must be one of revisions")
    if not (isinstance(c["stride_steps"], int) and c["stride_steps"] > 0):
        raise ValueError(f"{path.name} [pilot] stride_steps must be a positive int")
    if isinstance(c["seed"], bool) or not isinstance(c["seed"], int):
        raise ValueError(f"{path.name} [pilot] seed must be an int")
    rule = dict(c["rule"])
    tau_K, ladder = float(rule["tau_K"]), rule["tau_ladder"]
    if not (0 < tau_K < 1):
        raise ValueError(f"{path.name} [pilot.rule] tau_K must be in (0, 1)")
    if (not isinstance(ladder, list) or not ladder or float(ladder[0]) != tau_K
            or any(not (0 < float(t) <= tau_K) for t in ladder)
            or any(float(a) <= float(b) for a, b in zip(ladder, ladder[1:]))):
        raise ValueError(f"{path.name} [pilot.rule] tau_ladder must be strictly decreasing, start at tau_K, "
                         "and stay in (0, tau_K]: the ladder reads how far inside the tolerance f* sits")
    if not (0 < float(rule["holds_max"]) < float(rule["degrades_min"]) <= 1):
        raise ValueError(f"{path.name} [pilot.rule] needs 0 < holds_max < degrades_min <= 1")
    if not isinstance(c["pair"], str) or not c["pair"].strip() or "/" in c["pair"] or "\\" in c["pair"]:
        raise ValueError(f"{path.name} [pilot] pair must be a non-empty upstream Pair name without path separators")
    if (not isinstance(c["tokens"], str) or not c["tokens"].strip() or PurePosixPath(c["tokens"]).is_absolute()
            or PureWindowsPath(c["tokens"]).is_absolute()):
        raise ValueError(f"{path.name} [pilot] tokens must be a relative path under upstream_path")
    tsha = str(c["tokens_sha256"])
    if not (tsha.endswith("_PENDING") or (len(tsha) == 64 and all(ch in "0123456789abcdef" for ch in tsha))):
        raise ValueError(f"{path.name} [pilot] tokens_sha256 must be a 64-hex sha256 or a *_PENDING placeholder")
    ho = c["heldout"]
    if (not isinstance(ho, list) or len(ho) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in ho)
            or ho[0] < 0 or ho[1] <= ho[0]):
        raise ValueError(f"{path.name} [pilot] heldout must be [lo, hi) with 0 <= lo < hi")
    if c["dtype"] not in _DTYPES:
        raise ValueError(f"{path.name} [pilot] dtype must be one of {_DTYPES}")
    controls = dict(c["controls"])
    if not (0 <= float(controls["identity_max"]) < 1):
        raise ValueError(f"{path.name} [pilot.controls] identity_max must be in [0, 1)")
    if not (0 < float(controls["scrambled_min"]) <= 1):
        raise ValueError(f"{path.name} [pilot.controls] scrambled_min must be in (0, 1]")
    reg = str(c.get("registered_by", ""))
    if reg and not (len(reg) == 4 and reg.isdigit()):
        raise ValueError(f"{path.name} [pilot] registered_by must be a four-digit ledger entry or empty")
    root = Path(repo_root)
    return PilotConfig(
        source=str(c["source"]), base=str(c["base"]), revisions=revs, stride_steps=int(c["stride_steps"]),
        anchors=anchors, results_dir=root / c["results_dir"],
        upstream_path=(root / c["upstream_path"]).resolve(), upstream_sha=str(c["upstream_sha"]),
        pair=c["pair"], tokens=(root / c["upstream_path"]).resolve() / c["tokens"], tokens_sha256=tsha,
        heldout=(int(ho[0]), int(ho[1])), dtype=str(c["dtype"]),
        seed=int(c["seed"]), rule=rule, controls=controls, registered_by=reg, config_path=path,
    )
