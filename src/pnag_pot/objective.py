"""Penalized objective F and its closed-form gradient."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def _as_reg(regularizer):
    name = regularizer.lower().replace("-", "_")
    if name in ("quadratic", "quad"):
        return "quadratic"
    if name in ("elastic", "elastic_net", "elasticnet"):
        return "elastic_net"
    raise ValueError(f"unknown regularizer: {regularizer}")


def smoothness(regularizer, eta2=1.0):
    """Return (L, mu) of R.

    Quadratic: R = 0.5 * ||X||_F^2, so L = mu = 1 and grad R = X.
    Elastic net: R = (eta2/2) * ||X||_F^2 + eta1 * ||X||_1. The smooth
    part has L = mu = eta2. On X >= 0 the L1 term contributes the constant
    subgradient eta1.
    """
    kind = _as_reg(regularizer)
    if kind == "quadratic":
        return 1.0, 1.0
    return float(eta2), float(eta2)


def regularizer_value(X, regularizer="quadratic", eta1=0.0, eta2=1.0):
    """R(X). On X >= 0, ||X||_1 = sum(X), which is the form differentiated here."""
    X = jnp.asarray(X, dtype=jnp.float64)
    kind = _as_reg(regularizer)
    quad = 0.5 * jnp.sum(X ** 2)
    if kind == "quadratic":
        return quad
    # eta2/2 * ||X||_F^2 + eta1 * ||X||_1, with ||X||_1 = sum(X) for X >= 0.
    return float(eta2) * quad + float(eta1) * jnp.sum(X)


def exterior_penalty(X, r, c, alpha):
    """P(X, alpha) = alpha * sum of squared hinge violations of both marginals.

    The squared hinge min(0, z)^2 is C1, with derivative 2 * min(0, z).
    For a square plan the sum matches sum_i of the row hinge plus the column hinge.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    row = jnp.sum(X, axis=1)
    col = jnp.sum(X, axis=0)
    row_hinge = jnp.minimum(0.0, r - row) ** 2
    col_hinge = jnp.minimum(0.0, c - col) ** 2
    return float(alpha) * (jnp.sum(row_hinge) + jnp.sum(col_hinge))


def objective(X, C, r, c, eta, alpha, regularizer="quadratic", eta1=0.0, eta2=1.0):
    """F(X) = <C, X> + eta * R(X) + P(X, alpha)."""
    X = jnp.asarray(X, dtype=jnp.float64)
    C = jnp.asarray(C, dtype=jnp.float64)
    linear = jnp.sum(C * X)
    reg = regularizer_value(X, regularizer=regularizer, eta1=eta1, eta2=eta2)
    pen = exterior_penalty(X, r, c, alpha)
    return linear + float(eta) * reg + pen


def grad_F(X, C, r, c, eta, alpha, regularizer="quadratic", eta1=0.0, eta2=1.0):
    """Closed-form gradient of F, the penalized objective.

    This is not the gradient of <C, X> alone. The proximal map in PNAG-POT
    is only the projection onto S; R and the exterior penalty stay in F
    and are differentiated here.

    grad P_ij = -2 * alpha * (min(0, r_i - row_i) + min(0, c_j - col_j)).
    Quadratic: grad R = X.
    Elastic net, X >= 0: grad R_ij = eta2 * X_ij + eta1.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    C = jnp.asarray(C, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    row = jnp.sum(X, axis=1)
    col = jnp.sum(X, axis=0)
    u = jnp.minimum(0.0, r - row)
    v = jnp.minimum(0.0, c - col)
    grad_P = -2.0 * float(alpha) * (u[:, None] + v[None, :])
    kind = _as_reg(regularizer)
    if kind == "quadratic":
        grad_R = X
    else:
        # Nonnegative orthant subgradient of (eta2/2)||X||_F^2 + eta1||X||_1.
        grad_R = float(eta2) * X + float(eta1)
    return C + float(eta) * grad_R + grad_P
