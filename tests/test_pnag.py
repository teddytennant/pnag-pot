"""Gradient match, PNAG descent, LP comparison, elastic-net zeros."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from pnag_pot.objective import grad_F, objective
from pnag_pot.pnag import pnag_pot


def _in_U(X, r, c, s, tol=1e-5):
    X = np.asarray(X, dtype=np.float64)
    assert np.all(X >= -1e-7)
    assert np.all(X.sum(axis=1) <= np.asarray(r) + tol)
    assert np.all(X.sum(axis=0) <= np.asarray(c) + tol)
    assert abs(float(X.sum()) - s) < tol


def _lp_cost(C, r, c, s):
    """Unregularized partial OT cost via scipy.optimize.linprog, or None."""
    try:
        from scipy.optimize import linprog
    except ImportError:
        return None
    C = np.asarray(C, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    m, n = C.shape
    cost = C.reshape(-1)
    A_ub = []
    b_ub = []
    for i in range(m):
        row = np.zeros(m * n)
        row[i * n : (i + 1) * n] = 1.0
        A_ub.append(row)
        b_ub.append(r[i])
    for j in range(n):
        col = np.zeros(m * n)
        col[j::n] = 1.0
        A_ub.append(col)
        b_ub.append(c[j])
    res = linprog(
        cost,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.ones((1, m * n)),
        b_eq=np.array([s]),
        bounds=[(0.0, None)] * (m * n),
        method="highs",
    )
    assert res.success, res.message
    return float(res.fun)


def test_grad_F_matches_jax_grad_both_regularizers():
    rng = np.random.default_rng(2)
    C = jnp.asarray(rng.uniform(0.0, 1.0, size=(4, 4)))
    r = jnp.asarray(rng.uniform(0.5, 1.5, size=4))
    c = jnp.asarray(rng.uniform(0.5, 1.5, size=4))
    X = jnp.asarray(rng.normal(size=(4, 4)))
    # Include a boundary row so the squared hinge kink is exercised.
    X = X.at[0].set(r[0] / 4.0)

    for kind, eta1, eta2 in (("quadratic", 0.0, 1.0), ("elastic_net", 0.3, 0.7)):
        def F(Z, kind=kind, eta1=eta1, eta2=eta2):
            return objective(
                Z, C, r, c, eta=0.4, alpha=1.5, regularizer=kind, eta1=eta1, eta2=eta2
            )

        auto = jax.grad(F)(X)
        closed = grad_F(
            X, C, r, c, eta=0.4, alpha=1.5, regularizer=kind, eta1=eta1, eta2=eta2
        )
        err = float(jnp.max(jnp.abs(auto - closed)))
        assert err < 1e-5, (kind, err)


def test_pnag_descends_and_matches_lp_ballpark():
    rng = np.random.default_rng(3)
    n = 4
    C = rng.uniform(0.0, 1.0, size=(n, n))
    r = rng.uniform(0.8, 1.5, size=n)
    c = rng.uniform(0.8, 1.5, size=n)
    s = 0.6 * min(r.sum(), c.sum())
    alpha = 2.0
    steps = 60

    X_bar, info = pnag_pot(
        C,
        r,
        c,
        s,
        eta=0.05,
        alpha=alpha,
        regularizer="quadratic",
        n_steps=steps,
        return_trace=True,
    )
    F = [float(v) for v in info["F"]]
    assert F[-1] < F[0]
    _in_U(X_bar, r, c, s)

    lp = _lp_cost(C, r, c, s)
    if lp is None:
        # Hand-derived 2x2: cheapest entry is 1 and can take the whole mass.
        C = np.array([[1.0, 3.0], [4.0, 2.0]])
        r = np.array([1.0, 1.0])
        c = np.array([1.0, 1.0])
        s = 1.0
        lp = 1.0
        X_bar, info = pnag_pot(
            C, r, c, s, eta=1e-3, alpha=1.0, regularizer="quadratic", n_steps=60, return_trace=True
        )
        _in_U(X_bar, r, c, s)
        F = [float(v) for v in info["F"]]
        assert F[-1] < F[0]
    cost = float(np.sum(np.asarray(C) * np.asarray(X_bar)))
    # Feasible plans cannot beat the LP, up to solver tolerance.
    assert cost >= lp - 1e-4

    # Smaller eta must move the transport cost toward the LP cost.
    X_large = pnag_pot(
        C, r, c, s, eta=1.0, alpha=alpha, regularizer="quadratic", n_steps=steps
    )
    X_small = pnag_pot(
        C, r, c, s, eta=1e-4, alpha=alpha, regularizer="quadratic", n_steps=80
    )
    _in_U(X_large, r, c, s)
    _in_U(X_small, r, c, s)
    cost_large = float(np.sum(np.asarray(C) * np.asarray(X_large)))
    cost_small = float(np.sum(np.asarray(C) * np.asarray(X_small)))
    assert cost_large >= lp - 1e-4
    assert cost_small >= lp - 1e-4
    assert abs(cost_small - lp) <= abs(cost_large - lp) + 1e-3
    # Not wildly worse than the LP at small eta.
    assert cost_small <= lp + 0.25


def test_elastic_net_has_at_least_as_many_zeros_as_quadratic():
    rng = np.random.default_rng(4)
    n = 4
    C = rng.uniform(0.0, 1.0, size=(n, n))
    # Make one column clearly cheaper so L1 can concentrate.
    C[:, 0] *= 0.05
    r = np.full(n, 1.0)
    c = np.full(n, 1.0)
    s = 1.0
    common = dict(alpha=1.0, n_steps=60, eta=1.0, eta2=1.0)
    X_quad = np.asarray(
        pnag_pot(C, r, c, s, regularizer="quadratic", eta1=0.0, **common)
    )
    X_en = np.asarray(
        pnag_pot(C, r, c, s, regularizer="elastic_net", eta1=5.0, **common)
    )
    z_quad = int(np.sum(X_quad == 0.0))
    z_en = int(np.sum(X_en == 0.0))
    assert z_en >= z_quad
