"""Projection onto S = {X >= 0 : sum(X) = s}."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pnag_pot.projection import project_onto_S


def _kkt_lam(y, x, s, tol=1e-8):
    """Recover the simplex multiplier from positive entries: x_i = y_i + s * lam."""
    pos = np.asarray(x) > tol
    if not np.any(pos):
        raise AssertionError("projection has no positive entry")
    lam_vals = (np.asarray(x)[pos] - np.asarray(y)[pos]) / s
    assert np.allclose(lam_vals, lam_vals[0], atol=1e-6), lam_vals
    return float(lam_vals[0])


def test_projection_nonnegative_mass_and_kkt():
    rng = np.random.default_rng(0)
    for _ in range(20):
        shape = (rng.integers(2, 6), rng.integers(2, 6))
        Y = rng.normal(size=shape)
        # Force some large negative entries so the projection has zeros.
        Y.ravel()[0] = -5.0
        Y.ravel()[-1] = -3.0
        s = float(rng.uniform(0.2, 3.0))
        X = np.asarray(project_onto_S(Y, s))
        assert np.all(X >= -1e-10)
        assert abs(X.sum() - s) < 1e-6
        y = Y.ravel()
        x = X.ravel()
        lam = _kkt_lam(y, x, s)
        pos = x > 1e-8
        zero = ~pos
        assert np.allclose(x[pos], y[pos] + s * lam, atol=1e-6)
        assert np.all(y[zero] + s * lam <= 1e-6)


def test_wrong_lam_sign_fails_kkt():
    """Zero entries must satisfy y_i + s*lam <= 0. Negating lam fails that."""
    y = np.array([5.0, -1.0, -1.0])
    s = 1.0
    x = np.asarray(project_onto_S(y, s)).ravel()
    lam = _kkt_lam(y, x, s)
    assert lam < 0.0
    zero = x <= 1e-8
    pos = ~zero
    assert np.any(zero) and np.any(pos)
    # Plus sign is the projection KKT. The flipped multiplier is not.
    assert np.allclose(x[pos], y[pos] + s * lam, atol=1e-6)
    assert np.all(y[zero] + s * lam <= 1e-6)
    assert np.any(y[zero] + s * (-lam) > 1e-6)
    bad = np.maximum(y - s * lam, 0.0)
    assert not np.allclose(bad, x, atol=1e-6)
