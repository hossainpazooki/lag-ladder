"""The one seeded generator (invariant 4). Every function that needs randomness takes an
`rng: numpy.random.Generator` argument; nothing else in `src/` may construct one — tests
grep for it."""
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int from config, got {type(seed).__name__}")
    return np.random.default_rng(seed)
