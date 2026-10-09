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

# --- the weight-distance ladder (design §12, ledger 0007) ---------------------------------------------------

@dataclass(frozen=True)
class Ref:
    """One side of a pair as a dump sees it: `which` names the side, and at most one of `revision` (Hub) or
    `local_path` (a checkpoint directory under results_dir/checkpoints) pins it; neither means the Hub default."""
    which: str
    revision: str | None = None
    local_path: str | None = None


@dataclass(frozen=True)
class Rung:
    label: str
    pair: str
    writer: Ref
    reader: Ref
    kind: str = "real"        # real | noise
    rel_norm: float | None = None


@dataclass(frozen=True)
class PairTokens:
    name: str
    tokens: Path
    tokens_sha256: str


@dataclass(frozen=True)
class DistanceConfig:
    results_dir: Path
    upstream_path: Path
    upstream_sha: str
    heldout: tuple[int, int]
    dtype: str
    seed: int
    rule: dict
    controls: dict
    noise: dict               # pair, which, anchor_revision, rel_norms
    pairs: dict               # name -> PairTokens
    rungs: tuple[Rung, ...]   # the real rungs as written, then the noise rungs derived from [distance.noise]
    registered_by: str
    config_path: Path


def _plain_name(s, where: str) -> str:
    if not isinstance(s, str) or not s.strip() or "/" in s or "\\" in s or ".." in s:
        raise ValueError(f"{where} must be a plain name (no path separators)")
    return s


def _ref(d: dict, where: str) -> Ref:
    if not isinstance(d, dict) or d.get("which") not in ("source", "target"):
        raise ValueError(f"{where}: a ref needs which = source|target")
    rev, lp = d.get("revision"), d.get("local_path")
    if rev is not None and lp is not None:
        raise ValueError(f"{where}: a ref pins a revision or a local_path, not both")
    if rev is not None:
        _plain_name(rev, where + " revision")
    if lp is not None:
        _plain_name(lp, where + " local_path")
    return Ref(d["which"], rev, lp)


def _hex_or_pending(s, where: str) -> str:
    s = str(s)
    if not (s.endswith("_PENDING") or (len(s) == 64 and all(ch in "0123456789abcdef" for ch in s))):
        raise ValueError(f"{where} must be a 64-hex sha256 or a *_PENDING placeholder")
    return s


def _check_rule(rule: dict, where: str) -> dict:
    rule = dict(rule)
    tau_K, ladder = float(rule["tau_K"]), rule["tau_ladder"]
    if not (0 < tau_K < 1):
        raise ValueError(f"{where} tau_K must be in (0, 1)")
    if (not isinstance(ladder, list) or not ladder or float(ladder[0]) != tau_K
            or any(not (0 < float(t) <= tau_K) for t in ladder)
            or any(float(a) <= float(b) for a, b in zip(ladder, ladder[1:]))):
        raise ValueError(f"{where} tau_ladder must be strictly decreasing, start at tau_K, and stay in (0, tau_K]")
    if not (0 < float(rule["holds_max"]) < float(rule["degrades_min"]) <= 1):
        raise ValueError(f"{where} needs 0 < holds_max < degrades_min <= 1")
    return rule


def load_distance_config(path: Path, repo_root: Path) -> DistanceConfig:
    path = Path(path)
    c = _read(path)["distance"]
    for key in ("results_dir", "upstream_path", "upstream_sha", "heldout", "dtype", "seed", "rule", "controls",
                "noise", "pairs", "rungs"):
        if key not in c:
            raise ValueError(f"{path.name} [distance] is missing {key}")
    missing = [k for k in _RULE_KEYS if k not in c["rule"]]
    if missing:
        raise ValueError(f"{path.name} [distance.rule] is missing {missing}")
    missing = [k for k in _CONTROL_KEYS if k not in c["controls"]]
    if missing:
        raise ValueError(f"{path.name} [distance.controls] is missing {missing}")
    rule = _check_rule(c["rule"], f"{path.name} [distance.rule]")
    controls = dict(c["controls"])
    if not (0 <= float(controls["identity_max"]) < 1) or not (0 < float(controls["scrambled_min"]) <= 1):
        raise ValueError(f"{path.name} [distance.controls] identity_max in [0, 1), scrambled_min in (0, 1]")
    ho = c["heldout"]
    if (not isinstance(ho, list) or len(ho) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in ho)
            or ho[0] < 0 or ho[1] <= ho[0] + 1):
        raise ValueError(f"{path.name} [distance] heldout must be [lo, hi) with at least two sequences")
    if c["dtype"] not in _DTYPES:
        raise ValueError(f"{path.name} [distance] dtype must be one of {_DTYPES}")
    if isinstance(c["seed"], bool) or not isinstance(c["seed"], int):
        raise ValueError(f"{path.name} [distance] seed must be an int")
    root = Path(repo_root)
    upstream = (root / c["upstream_path"]).resolve()
    pairs = {}
    for i, pr in enumerate(c["pairs"]):
        where = f"{path.name} [[distance.pairs]] #{i}"
        for k in ("name", "tokens", "tokens_sha256"):
            if k not in pr:
                raise ValueError(f"{where} is missing {k}")
        name = _plain_name(pr["name"], where + " name")
        if name in pairs:
            raise ValueError(f"{where}: pair {name} listed twice")
        tok = pr["tokens"]
        if (not isinstance(tok, str) or not tok.strip() or PurePosixPath(tok).is_absolute()
                or PureWindowsPath(tok).is_absolute()):
            raise ValueError(f"{where}: tokens must be a relative path under upstream_path")
        pairs[name] = PairTokens(name, upstream / tok, _hex_or_pending(pr["tokens_sha256"], where + " tokens_sha256"))
    if not pairs:
        raise ValueError(f"{path.name} [distance] lists no pairs")
    noise = dict(c["noise"])
    for k in ("pair", "which", "anchor_revision", "rel_norms"):
        if k not in noise:
            raise ValueError(f"{path.name} [distance.noise] is missing {k}")
    if noise["pair"] not in pairs:
        raise ValueError(f"{path.name} [distance.noise] pair {noise['pair']!r} is not in [[distance.pairs]]")
    if noise["which"] not in ("source", "target"):
        raise ValueError(f"{path.name} [distance.noise] which must be source|target")
    _plain_name(noise["anchor_revision"], f"{path.name} [distance.noise] anchor_revision")
    norms = noise["rel_norms"]
    if (not isinstance(norms, list) or not norms or any(not (0 < float(n) < 1) for n in norms)
            or any(float(a) >= float(b) for a, b in zip(norms, norms[1:]))):
        raise ValueError(f"{path.name} [distance.noise] rel_norms must be strictly increasing and in (0, 1)")
    noise["rel_norms"] = [float(n) for n in norms]
    rungs, labels = [], set()
    for i, r in enumerate(c["rungs"]):
        where = f"{path.name} [[distance.rungs]] #{i}"
        for k in ("label", "pair", "writer", "reader"):
            if k not in r:
                raise ValueError(f"{where} is missing {k}")
        label = _plain_name(r["label"], where + " label")
        if label in labels or label.startswith("noise-"):
            raise ValueError(f"{where}: label {label!r} must be unique and not start with noise-")
        if r["pair"] not in pairs:
            raise ValueError(f"{where}: pair {r['pair']!r} is not in [[distance.pairs]]")
        w, rd = _ref(r["writer"], where + " writer"), _ref(r["reader"], where + " reader")
        if w == rd:
            raise ValueError(f"{where}: writer and reader are the same ref; identity is a control, not a rung")
        labels.add(label)
        rungs.append(Rung(label, r["pair"], w, rd))
    anchor = Ref(noise["which"], noise["anchor_revision"], None)
    for n in noise["rel_norms"]:
        label = f"noise-{n:g}"
        labels.add(label)
        rungs.append(Rung(label, noise["pair"], anchor, Ref(noise["which"], None, label), "noise", float(n)))
    reg = str(c.get("registered_by", ""))
    if reg and not (len(reg) == 4 and reg.isdigit()):
        raise ValueError(f"{path.name} [distance] registered_by must be a four-digit ledger entry or empty")
    return DistanceConfig(results_dir=root / c["results_dir"], upstream_path=upstream, upstream_sha=str(c["upstream_sha"]),
                          heldout=(int(ho[0]), int(ho[1])), dtype=str(c["dtype"]), seed=int(c["seed"]), rule=rule,
                          controls=controls, noise=noise, pairs=pairs, rungs=tuple(rungs), registered_by=reg,
                          config_path=path)
