import re
from pathlib import Path

import numpy as np
import pytest

from lag_ladder import REPO_ROOT
from lag_ladder.rng import make_rng

# Invariant 4: exactly one seeded generator (rng.py). These patterns cover both the
# direct call sites (default_rng(...), RandomState(...), np.random.seed(...),
# random.seed(...)) and the imports that would let a module reach those constructors
# under an alias that dodges the call-site substrings entirely, e.g.
#   from numpy.random import default_rng as _dr; _dr(seed)
_RNG_CONSTRUCTION_PATTERNS = (
    # direct call sites
    r"default_rng\(",
    r"RandomState\(",
    r"np\.random\.seed\(",
    r"random\.seed\(",
    # module-level imports that hand a file the means to construct a generator,
    # regardless of whether it's later called under an alias
    r"^\s*import\s+random\b",
    r"^\s*import\s+numpy\.random\b",
    r"^\s*from\s+numpy\s+import\s+.*\brandom\b",
    r"^\s*from\s+numpy\.random\s+import\s+.*\b(default_rng|Generator|PCG64|RandomState|seed)\b",
)
_RNG_CONSTRUCTION_RE = re.compile("|".join(_RNG_CONSTRUCTION_PATTERNS), re.MULTILINE)


def find_rng_construction_offenders(src_dir: Path) -> list[str]:
    """Return the names of .py files under src_dir that construct (or import the means
    to construct) a random generator, excluding rng.py itself."""
    offenders = []
    for p in src_dir.rglob("*.py"):
        if p.name == "rng.py":
            continue
        if _RNG_CONSTRUCTION_RE.search(p.read_text(encoding="utf-8")):
            offenders.append(p.name)
    return offenders


def test_same_seed_same_stream():
    a, b = make_rng(7), make_rng(7)
    assert np.array_equal(a.standard_normal(5), b.standard_normal(5))


def test_rejects_non_int_seed():
    with pytest.raises(TypeError):
        make_rng("0")
    with pytest.raises(TypeError):
        make_rng(True)


def test_only_rng_module_constructs_a_generator():
    """Invariant 4: one seeded generator. No other source file may call default_rng or RandomState."""
    assert find_rng_construction_offenders(REPO_ROOT / "src" / "lag_ladder") == []


def test_aliased_import_is_caught(tmp_path):
    """The old substring-only check missed `from numpy.random import default_rng as _dr`
    followed by `_dr(seed)` -- neither substring appears. Prove the hardened check catches it."""
    (tmp_path / "rng.py").write_text(
        "import numpy as np\n\ndef make_rng(seed):\n    return np.random.default_rng(seed)\n",
        encoding="utf-8",
    )
    sneaky = tmp_path / "sneaky.py"
    sneaky.write_text(
        "from numpy.random import default_rng as _dr\n\ndef build(seed):\n    return _dr(seed)\n",
        encoding="utf-8",
    )
    assert find_rng_construction_offenders(tmp_path) == ["sneaky.py"]
