"""MPGibbs latent smoother registry."""

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    _LATENT_SMOOTHER_DSMC,
    _LATENT_SMOOTHER_PLAIN,
)

from . import dsmc, plain_csmc

SMOOTHERS = {
    _LATENT_SMOOTHER_PLAIN: plain_csmc.smooth,
    _LATENT_SMOOTHER_DSMC: dsmc.smooth,
}
