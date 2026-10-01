"""PNAG-POT: proximal Nesterov on the penalized partial OT objective."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pnag_pot.objective import grad_F, objective, smoothness
from pnag_pot.projection import project_onto_S
from pnag_pot.round import round_pot


def _positive_theta(gamma, t, eta_mu):
    """Positive root of theta^2 / t = (1 - theta) * gamma + eta*mu*theta.

    Equivalent quadratic: theta^2 + t * (gamma - eta*mu) * theta - gamma * t = 0.
    The product of roots is negative, so exactly one root is positive.
    """
    gamma = float(gamma)
    t = float(t)
    eta_mu = float(eta_mu)
    b = t * (gamma - eta_mu)
    disc = b * b + 4.0 * gamma * t
    theta = (-b + disc ** 0.5) / 2.0
    return theta


def pnag_pot(
    C,
    r,
    c,
    s,
    *,
    eta=1.0,
    alpha=1.0,
    regularizer="quadratic",
    eta1=0.0,
    eta2=1.0,
    n_steps=50,
    X0=None,
    return_trace=False,
):
    """Run PNAG-POT and return ROUND-POT(X_k).

    The prox is projection onto S = {X >= 0 : sum(X) = s}. The gradient
    passed to that prox is grad F, the gradient of the penalized objective
    F = <C, X> + eta * R(X) + P(X, alpha), not the gradient of <C, X> alone.

    Step size t_i = 1 / beta with beta = eta * L + 4 * alpha * n, where n is
    the matrix order (max of the two dimensions; the paper's square n).
    L = mu = 1 for the quadratic regularizer and L = mu = eta2 for elastic net.

    theta_0 = 1 by choice. The recurrence needs a previous theta, and the
    i = 0 step is defined by Y = X_0 rather than by the extrapolation formula,
    so theta_0 is not taken from the quadratic. It seeds
    gamma_1 = theta_0^2 / t_0 and the V update. With this choice V_1 = X_1,
    the i = 1 gap V_1 - X_1 vanishes, and momentum starts at i = 2.

    For i >= 1,
        gamma_i = theta_{i-1}^2 / t_{i-1}
        theta_i > 0 solves theta_i^2 / t_i = (1 - theta_i) * gamma_i + eta*mu*theta_i
        Y = X_i + (theta_i * gamma_i) / (gamma_i + eta*mu*theta_i) * (V_i - X_i)
    Then
        X_{i+1} = proj_S(Y - t_i * grad_F(Y))
        V_{i+1} = X_i + (X_{i+1} - X_i) / theta_i
    After k steps the returned plan is ROUND-POT(X_k).
    """
    C = jnp.asarray(C, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    s = float(s)
    if s <= 0.0:
        raise ValueError("s must be positive")
    m, n_cols = C.shape
    # Paper beta uses the square order n. Rectangular plans use the larger side.
    n = max(m, n_cols)
    L, mu = smoothness(regularizer, eta2=eta2)
    beta = float(eta) * L + 4.0 * float(alpha) * n
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    t = 1.0 / beta
    eta_mu = float(eta) * mu

    if X0 is None:
        X = jnp.full((m, n_cols), s / (m * n_cols), dtype=jnp.float64)
    else:
        X = project_onto_S(jnp.asarray(X0, dtype=jnp.float64), s)
    X_init = X

    # theta_0 = 1. V_0 is never read; the i = 0 branch sets Y = X_0.
    theta_prev = 1.0
    t_prev = t
    V = X
    trace = [objective(X, C, r, c, eta, alpha, regularizer, eta1, eta2)]

    for i in range(int(n_steps)):
        t_i = t
        if i == 0:
            Y = X
            theta_i = theta_prev
        else:
            gamma_i = (theta_prev ** 2) / t_prev
            theta_i = _positive_theta(gamma_i, t_i, eta_mu)
            denom = gamma_i + eta_mu * theta_i
            coeff = (theta_i * gamma_i) / denom
            Y = X + coeff * (V - X)
        # grad of F (penalized objective), not grad of <C, X> alone.
        G = grad_F(Y, C, r, c, eta, alpha, regularizer, eta1, eta2)
        X_next = project_onto_S(Y - t_i * G, s)
        V_next = X + (X_next - X) / theta_i
        X = X_next
        V = V_next
        theta_prev = theta_i
        t_prev = t_i
        trace.append(objective(X, C, r, c, eta, alpha, regularizer, eta1, eta2))

    X_bar = round_pot(X, r, c, s)
    if return_trace:
        return X_bar, {"X0": X_init, "X_k": X, "F": trace, "theta": theta_prev}
    return X_bar
