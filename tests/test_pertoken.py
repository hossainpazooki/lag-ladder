"""The copied per-token statistics (linear-ceiling 0023): the exact bridge to 1 - R^2, f*, the band.
Tests copied from linear-ceiling tests/test_e9_pertoken.py at the chassis commit; the seam, block,
bootstrap and null-pairing tests are dropped with their functions."""
import numpy as np
import pytest

from lag_ladder.pertoken import band_outcome, centered_delta, f_star, layer_mean, token_mean


def test_centered_delta_token_mean_is_one_minus_r2_exactly():
    rng = np.random.default_rng(0)
    n, L, H = 40, 3, 2
    sq = rng.gamma(2.0, size=(n, L, H))
    sst = rng.uniform(50, 100, size=(L, H))
    d = centered_delta(sq, sst, n)
    r2 = 1 - sq.sum(0) / sst
    assert np.allclose(d.mean(0), 1 - r2, atol=1e-12)
    assert token_mean(d).shape == (n,) and layer_mean(d).shape == (n, L)
    assert token_mean(d).mean() == pytest.approx(1 - r2.mean())
    with pytest.raises(ValueError, match="not positive"):
        centered_delta(sq, np.zeros((L, H)), n)
    with pytest.raises(ValueError, match="shape mismatch"):
        centered_delta(sq, sst, n + 1)


def test_f_star_is_the_smallest_removed_fraction_and_zero_at_the_mean():
    d = np.array([0.1, 0.2, 0.3, 0.4, 5.0])
    assert f_star(d, d.mean()) == 0.0                      # tau = own mean -> nothing to remove
    assert f_star(d, 0.25) == pytest.approx(0.2)           # drop the 5.0: mean of the rest 0.25
    assert f_star(d, 0.2) == pytest.approx(0.4)            # drop 5.0 and 0.4: mean 0.2
    assert f_star(d, 0.0) == 1.0                           # nothing qualifies
    assert f_star(d, float(np.median(d))) > 0.0            # a median tau is NOT self-consistent
    assert f_star(np.zeros(3), 0.0) == 0.0
    with pytest.raises(ValueError):
        f_star(np.zeros(0), 0.1)
    with pytest.raises(ValueError):
        f_star(d, -1.0)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, -0.1])
def test_f_star_refuses_invalid_deviations(invalid):
    with pytest.raises(ValueError, match="finite and non-negative"):
        f_star([invalid, 0.1], 0.3186)


@pytest.mark.parametrize("invalid", [0.1, [[0.1], [0.2]]])
def test_f_star_refuses_non_vector_deviations(invalid):
    with pytest.raises(ValueError, match="one-dimensional"):
        f_star(invalid, 0.3186)


def test_band_outcome_edges():
    rule = {"holds_max": 0.15, "degrades_min": 0.50}
    assert band_outcome(0.15, rule) == "HOLDS" and band_outcome(0.0, rule) == "HOLDS"
    assert band_outcome(0.50, rule) == "DEGRADES" and band_outcome(1.0, rule) == "DEGRADES"
    assert band_outcome(0.3, rule) == "UNRESOLVED"
