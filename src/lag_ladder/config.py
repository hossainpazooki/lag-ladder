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
    heldout: tuple[int, int] | None = None     # §14: a pair's own held-out range (default: the config's)
    spans: Path | None = None                 # §14: per-sequence {"prefix_len": [...], "length": [...]} for stored continuations
    spans_sha256: str | None = None
    far_pair: str | None = None               # 0022: the pair whose dumps supply the far control's prefix (per_pair only)


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


_SCORE_TEXTS = ("corpus", "on-path", "rollout")   # rollout (§14): a stored continuation, per-rung only
_RULE_TEXTS = ("corpus", "on-path")
_GENERATE_ARMS = ("stale", "fresh")
_READER_RULE_KEYS = ("statistic", "holds_max", "degrades_min", "texts", "generate_under")
_READER_CONTROL_KEYS = ("identity_max_logit_diff", "scrambled_min")
_READER_FLOOR_KEYS = ("floor_max_median_kl",)


@dataclass(frozen=True)
class ReaderRung:
    """One statistic (B) cell (design §13): the reader model scores one continuation under the writer's
    dump (stale) against its own (fresh). Dumps are named by their key under
    results/distance/pairs/<pair>/dumps/ as the distance run wrote them; nothing is re-dumped."""
    label: str
    pair: str
    reader: str        # scorer spec: model_id[@revision] or local:<name> under KVT_CHECKPOINT_ROOT
    writer_dump: str   # dump key of the stale arm, e.g. "source@local-olmo2-1b-dpo"
    reader_dump: str   # dump key of the fresh arm (and of every control), e.g. "target@step_200"
    texts: tuple[str, ...] | None = None   # §14: this rung's texts when not the rule's (e.g. ("rollout",))


@dataclass(frozen=True)
class ReaderConfig:
    results_dir: Path
    upstream_path: Path
    upstream_sha: str
    heldout: tuple[int, int]
    prefix_len: int
    dtype: str
    seed: int
    rule: dict                # statistic, holds_max, degrades_min, texts, generate_under
    controls: dict            # identity_max_logit_diff, scrambled_min, floor_max_median_kl|None (0013: the floor halts on KL only)
    pairs: dict               # name -> PairTokens
    rungs: tuple[ReaderRung, ...]
    registered_by: str
    config_path: Path
    seal: str = "reader"      # the sealed prediction this config runs against (ledger/predictions/<seal>.json)


def _dump_key(s, where: str) -> str:
    if (not isinstance(s, str) or "@" not in s or "/" in s or "\\" in s or ".." in s
            or not all(part.strip() for part in s.split("@", 1))):
        raise ValueError(f"{where} must be a dump key <side>@<pin> (no path separators)")
    return s


def load_reader_config(path: Path, repo_root: Path) -> ReaderConfig:
    path = Path(path)
    c = _read(path)["reader"]
    for key in ("results_dir", "upstream_path", "upstream_sha", "heldout", "prefix_len", "dtype", "seed",
                "rule", "controls", "pairs", "rungs"):
        if key not in c:
            raise ValueError(f"{path.name} [reader] is missing {key}")
    missing = [k for k in _READER_RULE_KEYS if k not in c["rule"]]
    if missing:
        raise ValueError(f"{path.name} [reader.rule] is missing {missing}")
    missing = [k for k in _READER_CONTROL_KEYS if k not in c["controls"]]
    if missing:
        raise ValueError(f"{path.name} [reader.controls] is missing {missing}")
    rule = dict(c["rule"])
    if not (0 < float(rule["holds_max"]) < float(rule["degrades_min"])):
        raise ValueError(f"{path.name} [reader.rule] needs 0 < holds_max < degrades_min")
    texts = rule["texts"]
    if (not isinstance(texts, list) or not texts or len(set(texts)) != len(texts)
            or any(t not in _RULE_TEXTS for t in texts)):
        raise ValueError(f"{path.name} [reader.rule] texts must be distinct entries from {_RULE_TEXTS}; "
                         "rollout is a per-rung text")
    if rule["generate_under"] not in _GENERATE_ARMS:
        raise ValueError(f"{path.name} [reader.rule] generate_under must be one of {_GENERATE_ARMS}")
    controls = dict(c["controls"])
    if not (0 < float(controls["identity_max_logit_diff"]) < 1):
        raise ValueError(f"{path.name} [reader.controls] identity_max_logit_diff must be in (0, 1)")
    if not (0 < float(controls["scrambled_min"])):
        raise ValueError(f"{path.name} [reader.controls] scrambled_min must be positive")
    # 0020: controls per (reader, pair) on the pair's own texts; the default keeps 0012-0018's plans as they ran.
    per_pair = controls.get("per_pair", False)
    if not isinstance(per_pair, bool):
        raise ValueError(f"{path.name} [reader.controls] per_pair must be a bool when set")
    controls["per_pair"] = per_pair
    # 0022: the far control's limit; required when any pair names a far_pair (validated after the pairs are read).
    if "far_min" in controls and not (float(controls["far_min"]) > 0):
        raise ValueError(f"{path.name} [reader.controls] far_min must be positive")
    for k in _READER_FLOOR_KEYS:
        # Empty-able: 0012 fixes them from the floor probe; `run` refuses while they are unset.
        if k not in controls:
            controls[k] = None
        elif not (float(controls[k]) > 0):
            raise ValueError(f"{path.name} [reader.controls] {k} must be a positive float when set")
    def _heldout(ho, where):
        if (not isinstance(ho, list) or len(ho) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in ho)
                or ho[0] < 0 or ho[1] <= ho[0] + 1):
            raise ValueError(f"{where} heldout must be [lo, hi) with at least two sequences")
        return (int(ho[0]), int(ho[1]))

    ho = _heldout(c["heldout"], f"{path.name} [reader]")
    pl = c["prefix_len"]
    if isinstance(pl, bool) or not isinstance(pl, int) or pl <= 0:
        raise ValueError(f"{path.name} [reader] prefix_len must be a positive int")
    if c["dtype"] not in _DTYPES:
        raise ValueError(f"{path.name} [reader] dtype must be one of {_DTYPES}")
    if isinstance(c["seed"], bool) or not isinstance(c["seed"], int):
        raise ValueError(f"{path.name} [reader] seed must be an int")
    root = Path(repo_root)
    upstream = (root / c["upstream_path"]).resolve()
    pairs = {}
    for i, pr in enumerate(c["pairs"]):
        where = f"{path.name} [[reader.pairs]] #{i}"
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
        pho = _heldout(pr["heldout"], where) if "heldout" in pr else None
        spans = spans_sha = None
        if "spans" in pr or "spans_sha256" in pr:
            sp = pr.get("spans")
            if (not isinstance(sp, str) or not sp.strip() or PurePosixPath(sp).is_absolute()
                    or PureWindowsPath(sp).is_absolute() or "spans_sha256" not in pr):
                raise ValueError(f"{where}: spans must be a relative path under upstream_path, with spans_sha256")
            spans, spans_sha = upstream / sp, _hex_or_pending(pr["spans_sha256"], where + " spans_sha256")
        far = pr.get("far_pair")
        if far is not None and (not isinstance(far, str) or not far.strip()):
            raise ValueError(f"{where}: far_pair must name a pair")
        pairs[name] = PairTokens(name, upstream / tok, _hex_or_pending(pr["tokens_sha256"], where + " tokens_sha256"),
                                 pho, spans, spans_sha, far)
    if not pairs:
        raise ValueError(f"{path.name} [reader] lists no pairs")
    for pt in pairs.values():
        if pt.far_pair is None:
            continue
        if pt.far_pair not in pairs or pt.far_pair == pt.name:
            raise ValueError(f"{path.name} [[reader.pairs]] {pt.name}: far_pair {pt.far_pair!r} must name another pair in the table")
        if not per_pair:
            raise ValueError(f"{path.name} [[reader.pairs]] {pt.name}: far_pair needs [reader.controls] per_pair = true")
        if "far_min" not in controls:
            raise ValueError(f"{path.name} [reader.controls] far_min is required when a pair names a far_pair")
    rungs, labels = [], set()
    for i, r in enumerate(c["rungs"]):
        where = f"{path.name} [[reader.rungs]] #{i}"
        for k in ("label", "pair", "reader", "writer_dump", "reader_dump"):
            if k not in r:
                raise ValueError(f"{where} is missing {k}")
        label = _plain_name(r["label"], where + " label")
        if label in labels:
            raise ValueError(f"{where}: label {label!r} must be unique")
        if r["pair"] not in pairs:
            raise ValueError(f"{where}: pair {r['pair']!r} is not in [[reader.pairs]]")
        if not isinstance(r["reader"], str) or not r["reader"].strip():
            raise ValueError(f"{where}: reader must be a scorer spec (model_id[@revision] or local:<name>)")
        w = _dump_key(r["writer_dump"], where + " writer_dump")
        rd = _dump_key(r["reader_dump"], where + " reader_dump")
        if w == rd:
            raise ValueError(f"{where}: writer_dump and reader_dump are the same dump; identity is a control, not a rung")
        rtexts = None
        if "texts" in r:
            rt = r["texts"]
            if (not isinstance(rt, list) or not rt or len(set(rt)) != len(rt) or any(t not in _SCORE_TEXTS for t in rt)):
                raise ValueError(f"{where}: texts must be distinct entries from {_SCORE_TEXTS}")
            if "rollout" in rt and pairs[r["pair"]].spans is None:
                raise ValueError(f"{where}: a rollout text needs a pair with spans (per-sequence prefix_len and length)")
            rtexts = tuple(str(t) for t in rt)
        labels.add(label)
        rungs.append(ReaderRung(label, r["pair"], str(r["reader"]), w, rd, rtexts))
    if not rungs:
        raise ValueError(f"{path.name} [reader] lists no rungs")
    first = {}
    for r in rungs:
        first.setdefault(r.reader, r)
    for r in ([] if controls["per_pair"] else first.values()):
        if r.texts is not None and "rollout" in r.texts:
            raise ValueError(f"{path.name}: reader {r.reader}'s first rung ({r.label}) is a rollout rung; its controls "
                             "run on the first rung's pair, so a corpus rung must come first")
    reg = str(c.get("registered_by", ""))
    if reg and not (len(reg) == 4 and reg.isdigit()):
        raise ValueError(f"{path.name} [reader] registered_by must be a four-digit ledger entry or empty")
    return ReaderConfig(results_dir=root / c["results_dir"], upstream_path=upstream,
                        upstream_sha=str(c["upstream_sha"]), heldout=(int(ho[0]), int(ho[1])),
                        prefix_len=int(pl), dtype=str(c["dtype"]), seed=int(c["seed"]), rule=rule,
                        controls=controls, pairs=pairs, rungs=tuple(rungs), registered_by=reg, config_path=path,
                        seal=_plain_name(c.get("seal", "reader"), f"{path.name} [reader] seal"))


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


# --- statistic (C): the task driver's configuration (design §15) -------------------------------------------------

_TASK_RULE_KEYS = ("holds_abs_max", "holds_ci_abs_max", "degrades_max", "degrades_ci_max", "bootstrap_reps",
                   "control_min_flip_rate")
_TASK_CONTROL_KEYS = ("n_prompts", "floor_max_flip_rate", "equivalence_max_rel_diff", "equivalence_b1_max_rel",
                      "equivalence_reader", "equivalence_expect_b1", "rollout_tokens", "rollout_tokens_sha256",
                      "rollout_spans", "rollout_spans_sha256", "rollout_dump_root")


@dataclass(frozen=True)
class TaskConfig:
    results_dir: Path
    upstream_path: Path
    upstream_sha: str
    registered_by: str
    seal: str
    writer: str               # the step-0 policy: local:<name> under results/distance/checkpoints
    readers: tuple[str, ...]  # local:step_<g>
    prompts: Path             # [{index, question, gold}] under upstream data/tokens
    prompts_sha256: str
    n_prompts: int
    far_tokens: Path
    far_tokens_sha256: str
    far_seq: int
    max_new: int
    seed: int
    rule: dict
    controls: dict            # paths resolved under upstream; the rest as written
    config_path: Path


def _rel_under_upstream(s, where: str, upstream: Path) -> Path:
    if (not isinstance(s, str) or not s.strip() or PurePosixPath(s).is_absolute() or PureWindowsPath(s).is_absolute()
            or ".." in PurePosixPath(s).parts):
        raise ValueError(f"{where} must be a relative path under upstream_path")
    return upstream / s


def _local_spec(s, where: str) -> str:
    if not isinstance(s, str) or not s.startswith("local:") or not s[len("local:"):].strip():
        raise ValueError(f"{where} must be a local:<name> checkpoint spec")
    _plain_name(s[len("local:"):], where)
    return s


def load_task_config(path: Path, repo_root: Path) -> TaskConfig:
    path = Path(path)
    with open(path, "rb") as f:
        c = tomllib.load(f)["task"]
    for key in ("results_dir", "upstream_path", "upstream_sha", "writer", "readers", "prompts", "prompts_sha256",
                "n_prompts", "far_tokens", "far_tokens_sha256", "far_seq", "max_new", "seed", "rule", "controls"):
        if key not in c:
            raise ValueError(f"{path.name} [task] is missing {key}")
    root = Path(repo_root)
    upstream = (root / c["upstream_path"]).resolve()
    writer = _local_spec(c["writer"], f"{path.name} [task] writer")
    readers = c["readers"]
    if not isinstance(readers, list) or not readers or len(set(readers)) != len(readers):
        raise ValueError(f"{path.name} [task] readers must be a non-empty list of distinct specs")
    readers = tuple(_local_spec(r, f"{path.name} [task] readers") for r in readers)
    if writer in readers:
        raise ValueError(f"{path.name} [task] writer {writer} is also a reader; FRESH vs STALE would be identity")
    n = int(c["n_prompts"])
    if n <= 0:
        raise ValueError(f"{path.name} [task] n_prompts must be positive")
    if int(c["max_new"]) <= 0 or int(c["far_seq"]) < 0:
        raise ValueError(f"{path.name} [task] max_new must be positive and far_seq non-negative")
    rule = dict(c["rule"])
    for k in _TASK_RULE_KEYS:
        if k not in rule:
            raise ValueError(f"{path.name} [task.rule] is missing {k}")
    if not (0 < float(rule["holds_abs_max"]) <= float(rule["holds_ci_abs_max"]) < 5):
        raise ValueError(f"{path.name} [task.rule] needs 0 < holds_abs_max <= holds_ci_abs_max < 5 (points)")
    if not (float(rule["degrades_max"]) <= float(rule["degrades_ci_max"]) < 0):
        raise ValueError(f"{path.name} [task.rule] needs degrades_max <= degrades_ci_max < 0 (points)")
    if not (0 < float(rule["control_min_flip_rate"]) <= 1):
        raise ValueError(f"{path.name} [task.rule] control_min_flip_rate must be in (0, 1]")
    if int(rule["bootstrap_reps"]) < 100:
        raise ValueError(f"{path.name} [task.rule] bootstrap_reps must be at least 100")
    controls = dict(c["controls"])
    for k in _TASK_CONTROL_KEYS:
        if k not in controls:
            raise ValueError(f"{path.name} [task.controls] is missing {k}")
    if not (0 < int(controls["n_prompts"]) <= n):
        raise ValueError(f"{path.name} [task.controls] n_prompts must be in (0, task.n_prompts]")
    if not (0 <= float(controls["floor_max_flip_rate"]) < 1):
        raise ValueError(f"{path.name} [task.controls] floor_max_flip_rate must be in [0, 1)")
    if not (0 <= float(controls["equivalence_max_rel_diff"]) < 1) or not (0 < float(controls["equivalence_b1_max_rel"]) < 1):
        raise ValueError(f"{path.name} [task.controls] equivalence limits out of range")
    if float(controls["equivalence_expect_b1"]) <= 0:
        raise ValueError(f"{path.name} [task.controls] equivalence_expect_b1 must be positive")
    controls["equivalence_reader"] = _local_spec(controls["equivalence_reader"], f"{path.name} [task.controls] equivalence_reader")
    controls["rollout_tokens"] = _rel_under_upstream(controls["rollout_tokens"], f"{path.name} [task.controls] rollout_tokens", upstream)
    controls["rollout_spans"] = _rel_under_upstream(controls["rollout_spans"], f"{path.name} [task.controls] rollout_spans", upstream)
    controls["rollout_tokens_sha256"] = _hex_or_pending(controls["rollout_tokens_sha256"], f"{path.name} rollout_tokens_sha256")
    controls["rollout_spans_sha256"] = _hex_or_pending(controls["rollout_spans_sha256"], f"{path.name} rollout_spans_sha256")
    dr = controls["rollout_dump_root"]
    if not isinstance(dr, str) or PurePosixPath(dr).is_absolute() or PureWindowsPath(dr).is_absolute() or ".." in PurePosixPath(dr).parts:
        raise ValueError(f"{path.name} [task.controls] rollout_dump_root must be a relative path under the repository")
    controls["rollout_dump_root"] = root / dr
    return TaskConfig(
        results_dir=root / c["results_dir"], upstream_path=upstream, upstream_sha=str(c["upstream_sha"]),
        registered_by=str(c.get("registered_by", "")), seal=str(c.get("seal", "ownrun-task")),
        writer=writer, readers=readers,
        prompts=_rel_under_upstream(c["prompts"], f"{path.name} [task] prompts", upstream),
        prompts_sha256=_hex_or_pending(c["prompts_sha256"], f"{path.name} prompts_sha256"), n_prompts=n,
        far_tokens=_rel_under_upstream(c["far_tokens"], f"{path.name} [task] far_tokens", upstream),
        far_tokens_sha256=_hex_or_pending(c["far_tokens_sha256"], f"{path.name} far_tokens_sha256"),
        far_seq=int(c["far_seq"]), max_new=int(c["max_new"]), seed=int(c["seed"]), rule=rule, controls=controls,
        config_path=path)
