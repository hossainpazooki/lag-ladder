"""lag-ladder: the ledger, seal and gate tooling for Holdover — a KV cache held over a weight update.

Nothing here fits, dumps or scores a model. The instrument lives in the pinned upstream repository
and is invoked, never imported; the chassis is copied from linear-ceiling with provenance (UPSTREAM.md).
"""
from pathlib import Path

__version__ = "0.0.1"

# src/lag_ladder/__init__.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Single source of truth for every pin; UPSTREAM.md repeats them for humans and tests/test_imports.py
# asserts the two agree (every 40-hex sha in UPSTREAM.md is one of these, and each of these is there).
INSTRUMENT_REPO = "https://github.com/hossainpazooki/kv-transfer-replication"
INSTRUMENT_SHA = "063f4023fdde67dedbee01a92518ce7f83f6cf5d"   # HEAD on 2026-09-22 = linear-ceiling's 0036 pin; re-pin by the entry that lands the Pair revision field

CHASSIS_REPO = "https://github.com/hossainpazooki/linear-ceiling"
CHASSIS_SHA = "888f745084c63eb52d114acd951dc787db82a71a"      # the commit the copied modules were read at (src/ clean at that HEAD, 2026-09-22)

PRODUCER_REPO = "https://github.com/ServiceNow/PipelineRL"
PRODUCER_OBSERVED_SHA = "58d393458625ad63ed539f2dcd072c85700c557f"   # main on 2026-09-22; OBSERVED, NOT PINNED (design §6: producer deferred)
