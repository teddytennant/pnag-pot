"""Exterior penalty and ROUND-POT."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pnag_pot.objective import exterior_penalty
from pnag_pot.round import round_pot


def _in_U(X, r, c, s, tol=1e-5):
    X = np.asarray(X)
    assert np.all(X >= -1e-8)
    assert np.all(np.asarray(X).sum(axis=1) <= np.asarray(r) + tol)
    assert np.all(np.asarray(X).sum(axis=0) <= np.asarray(c) + tol)
    assert abs(float(np.asarray(X).sum()) - s) < tol


def test_penalty_zero_positive_and_monotone_in_alpha():
    r = jnp.array([1.0, 1.0, 1.0])
    c = jnp.array([1.0, 1.0])
    # Row sums [0.5, 0.4, 0.3] and col sums [0.6, 0.6] respect both marginals.
    X_ok = jnp.array([[0.3, 0.2], [0.2, 0.2], [0.1, 0.2]])
    assert float(exterior_penalty(X_ok, r, c, alpha=3.0)) == 0.0

    # First row sum is 1.5 > r_0 = 1.
    X_bad = jnp.array([[1.0, 0.5], [0.1, 0.1], [0.0, 0.0]])
    p1 = float(exterior_penalty(X_bad, r, c, alpha=1.0))
    p2 = float(exterior_penalty(X_bad, r, c, alpha=2.5))
    assert p1 > 0.0
    assert p2 > p1


def test_round_pot_repairs_violating_nonnegative_matrix():
    rng = np.random.default_rng(1)
    for _ in range(15):
        m, n = 4, 3
        r = rng.uniform(0.5, 2.0, size=m)
        c = rng.uniform(0.5, 2.0, size=n)
        s = 0.5 * min(r.sum(), c.sum())
        X = rng.uniform(0.0, 1.5, size=(m, n))
        # Guarantee a marginal violation.
        X[0, :] = r[0] + 0.7
        X_bar = round_pot(X, r, c, s)
        _in_U(X_bar, r, c, s)


def test_round_pot_keeps_feasible_input_feasible():
    r = np.array([1.2, 0.8, 1.0])
    c = np.array([1.0, 1.5])
    s = 1.1
    # Feasible plan: nonnegative, marginals respected, mass s.
    X = np.array([[0.4, 0.2], [0.1, 0.2], [0.1, 0.1]])
    assert abs(X.sum() - s) < 1e-12
    assert np.all(X.sum(axis=1) <= r + 1e-12)
    assert np.all(X.sum(axis=0) <= c + 1e-12)
    X_bar = np.asarray(round_pot(X, r, c, s))
    _in_U(X_bar, r, c, s, tol=1e-5)
