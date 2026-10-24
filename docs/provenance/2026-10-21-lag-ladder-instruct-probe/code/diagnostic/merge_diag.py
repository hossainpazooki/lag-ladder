"""Merge one cap's shard folders into a single probe folder (records.jsonl in index order + probe.json), using the
package's own summarize() so every field means what it means in the registered probe.json.

    python -I -B merge_diag.py <package dir> <lag-ladder dir> <cap> <n> <out dir> <shard dir> [<shard dir> ...]
Prints MERGE_OK <cap> or MERGE_FAIL <reason> and exits nonzero on failure.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
pkg, ll, cap, n, out = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), Path(sys.argv[5])
shards = [Path(p) for p in sys.argv[6:]]
sys.path.insert(0, pkg)
import baseline_probe   # noqa: E402


def fail(msg):
    print(f"MERGE_FAIL cap {cap}: {msg}")
    raise SystemExit(1)


records, envs, elapsed, starts = [], [], [], []
for s in shards:
    p = json.loads((s / "probe.json").read_text(encoding="utf-8"))
    if p["max_new"] != cap:
        fail(f"{s} max_new {p['max_new']}")
    recs = [json.loads(l) for l in (s / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    if len(recs) != p["n"]:
        fail(f"{s} has {len(recs)} records, probe.json says {p['n']}")
    records += recs; envs.append(p["env"]); elapsed.append(p["elapsed_s"]); starts.append(p["start"])
records.sort(key=lambda r: r["index"])
idx = [r["index"] for r in records]
if idx != list(range(n)):
    fail(f"indices are not exactly 0..{n - 1} (got {len(idx)}, {len(set(idx))} unique)")
if any(e != envs[0] for e in envs):
    fail("shards ran in different environments")

SYSTEM_LINE = baseline_probe.load_rules(ll)[0]
prompts_sha = hashlib.sha256(Path(pkg, "gsm8k_test_prompts.json").read_bytes()).hexdigest()
out.mkdir(parents=True, exist_ok=False)
rec_path = out / "records.jsonl"
with rec_path.open("w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
s = baseline_probe.summarize(records)
s.update({"schema": "baseline-probe-summary-v1", "prompts_sha256": prompts_sha, "start": 0, "max_new": cap,
          "system_line": SYSTEM_LINE, "env": envs[0], "elapsed_s": max(elapsed),
          "records_sha256": hashlib.sha256(rec_path.read_bytes()).hexdigest(),
          "diagnostic": {"what": "NOT the registered probe. Decoding = plain argmax, repetition_penalty 1.0, stop only on "
                                 "tok.eos_token_id (the registered (C) loop's decoding); everything else as baseline_probe.py.",
                         "shards": len(shards), "shard_starts": sorted(starts),
                         "elapsed_s_note": "max over shards run in parallel, one card each"}})
(out / "probe.json").write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
print(f"MERGE_OK cap {cap} n {s['n']} accuracy {s['accuracy_points']:.1f} [{s['accuracy_ci95'][0]:.1f}, "
      f"{s['accuracy_ci95'][1]:.1f}] cap {s['cap_rate']:.2f} no_answer {s['no_answer_rate']:.2f} "
      f"hash_marker {1 - s['no_hash_marker_rate']:.2f} boxed_marker {s['boxed_marker_rate']:.2f} "
      f"acc_boxed {s['accuracy_boxed_points']:.1f} loop {s['first_vs_last_hash_disagree_rate']:.2f} band {s['band']}")
