"""MPGibbs latent smoother registry."""

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    _LATENT_SMOOTHER_DSMC,
)

from . import dsmc

SMOOTHERS = {
    _LATENT_SMOOTHER_DSMC: dsmc.smooth,
}
