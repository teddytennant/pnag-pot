"""Euclidean projection onto S = {X >= 0 : sum(X) = s}."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def project_onto_S(Y, s):
    """Project Y onto the scaled simplex of mass s.

    Duchi's probability-simplex projection is applied to Y / s, then the
    result is multiplied by s. With u the descending sort of y / s,

        rho = max {j : u_j + (1/j) * (1 - sum_{i=1..j} u_i) > 0}
        lam = (1/rho) * (1 - sum_{i=1..rho} u_i)
        x' = max(y / s + lam, 0)
        x = s * x'

    Positive entries of x equal y_i + s * lam. Zero entries have
    y_i + s * lam <= 0. The sign of lam is the Duchi sign above; flipping
    it does not yield the Euclidean projection.
    """
    Y = jnp.asarray(Y, dtype=jnp.float64)
    s = jnp.asarray(s, dtype=jnp.float64)
    if float(s) <= 0.0:
        raise ValueError("s must be positive")
    y = jnp.reshape(Y, (-1,))
    y_tilde = y / s
    # Descending sort. rho is at least 1 because u_1 + (1 - u_1) = 1 > 0.
    u = jnp.sort(y_tilde)[::-1]
    n = y.shape[0]
    j = jnp.arange(1, n + 1, dtype=jnp.float64)
    cssv = jnp.cumsum(u)
    cond = u + (1.0 / j) * (1.0 - cssv) > 0.0
    rho = jnp.max(jnp.where(cond, j, 0.0))
    rho_idx = (rho - 1.0).astype(jnp.int32)
    lam = (1.0 / rho) * (1.0 - cssv[rho_idx])
    x_prime = jnp.maximum(y_tilde + lam, 0.0)
    x = s * x_prime
    return jnp.reshape(x, Y.shape)
