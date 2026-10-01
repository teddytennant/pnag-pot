"""EP slack and ROUND-POT rounding into the partial transport set U."""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


def _walk_marginal(marginal, budget):
    """Greedy index walk: fill marginal capacity until the sum equals budget.

    Walk i until the running sum would exceed the budget, put the residual
    on that coordinate, and zero the rest.
    """
    m = np.asarray(marginal, dtype=np.float64)
    out = np.zeros_like(m)
    running = 0.0
    target = float(budget)
    if target <= 0.0:
        return jnp.asarray(out)
    for i in range(m.shape[0]):
        room = float(m[i])
        if room < 0.0:
            room = 0.0
        if running + room > target:
            out[i] = target - running
            running = target
            break
        out[i] = room
        running += room
    return jnp.asarray(out)


def _top_up(base, marginal, budget):
    """Keep base slack, then walk remaining marginal capacity up to budget."""
    out = np.asarray(base, dtype=np.float64).copy()
    cap = np.asarray(marginal, dtype=np.float64)
    running = float(out.sum())
    target = float(budget)
    for i in range(out.shape[0]):
        if running >= target - 1e-15:
            break
        room = float(cap[i]) - float(out[i])
        if room <= 0.0:
            continue
        take = min(room, target - running)
        out[i] = float(out[i]) + take
        running += take
    return jnp.asarray(out)


def ep(marginal, s, slack):
    """Exterior-penalty slack EP(marginal, s, p).

    p' = min(p, marginal). If ||p'||_1 == 0 and the residual budget
    ||marginal||_1 - s is not positive, return zeros (the spec case).
    psi = min(1, budget / ||p'||_1), p'' = psi * p'.
    If psi < 1, return p''. Otherwise the capped slack does not exceed the
    budget: keep p' when it already meets the budget (the index walk does
    not truncate it), and if it falls short, walk marginal capacity so the
    returned slack sums to ||marginal||_1 - s and stays componentwise at most
    the marginal. Returning literal zeros when p' vanishes but the budget is
    positive cannot put ROUND-POT in U, so that case uses the same walk.
    """
    marginal = jnp.asarray(marginal, dtype=jnp.float64)
    slack = jnp.asarray(slack, dtype=jnp.float64)
    budget = float(jnp.sum(marginal) - s)
    if budget <= 1e-12:
        return jnp.zeros_like(marginal)
    p_prime = jnp.minimum(jnp.maximum(slack, 0.0), jnp.maximum(marginal, 0.0))
    sp = float(jnp.sum(p_prime))
    if sp <= 1e-15:
        return _walk_marginal(marginal, budget)
    psi = min(1.0, budget / sp)
    p_double = psi * p_prime
    if psi < 1.0:
        return p_double
    if sp >= budget - 1e-8:
        return p_prime
    return _top_up(p_prime, marginal, budget)


def _row_scale(target, mass):
    """min(1, target / mass), with 0/0 := 0.

    A zero mass row is left at scale 0. Scaling it by 1 would not change the
    product with that row, and the spec's 0/0 rule returns 0.
    """
    target = jnp.asarray(target, dtype=jnp.float64)
    mass = jnp.asarray(mass, dtype=jnp.float64)
    scale = jnp.where(
        mass > 0.0,
        jnp.minimum(1.0, target / mass),
        0.0,
    )
    return jnp.clip(scale, 0.0, 1.0)


def round_pot(X, r, c, s):
    """ROUND-POT(X, r, c, s). The result lies in U for nonnegative X.

    U = {X >= 0 : X 1 <= r, X^T 1 <= c, sum(X) = s}, assuming
    0 < s <= min(sum(r), sum(c)).
    """
    X = jnp.maximum(jnp.asarray(X, dtype=jnp.float64), 0.0)
    r = jnp.asarray(r, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    s = float(s)
    row = jnp.sum(X, axis=1)
    col = jnp.sum(X, axis=0)
    p = jnp.maximum(0.0, r - row)
    q = jnp.maximum(0.0, c - col)
    p_bar = ep(r, s, p)
    q_bar = ep(c, s, q)
    r_hat = r - p_bar
    c_hat = c - q_bar
    g = _row_scale(r_hat, row)
    h = _row_scale(c_hat, col)
    X_prime = g[:, None] * X * h[None, :]
    e1 = r_hat - jnp.sum(X_prime, axis=1)
    e2 = c_hat - jnp.sum(X_prime, axis=0)
    # Spec divides by the L1 norm. For the nonnegative residuals produced
    # above this equals the signed sum, which is what keeps both marginals.
    e1_l1 = jnp.sum(jnp.abs(e1))
    if float(e1_l1) == 0.0:
        X_bar = X_prime
    else:
        X_bar = X_prime + jnp.outer(e1, e2) / e1_l1
    # Rank-one fill-in leaves 1e-16 dust on structural zeros. Drop it so
    # those zeros stay exact; the mass change is far below the U tolerance.
    X_bar = jnp.where(jnp.abs(X_bar) < 1e-12, 0.0, X_bar)
    return jnp.maximum(X_bar, 0.0)
