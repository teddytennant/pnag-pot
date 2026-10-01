"""Proximal Nesterov for regularized partial optimal transport."""

import jax

jax.config.update("jax_enable_x64", True)

from pnag_pot.objective import exterior_penalty, grad_F, objective
from pnag_pot.pnag import pnag_pot
from pnag_pot.projection import project_onto_S
from pnag_pot.round import ep, round_pot

__all__ = [
    "ep",
    "exterior_penalty",
    "grad_F",
    "objective",
    "pnag_pot",
    "project_onto_S",
    "round_pot",
]
