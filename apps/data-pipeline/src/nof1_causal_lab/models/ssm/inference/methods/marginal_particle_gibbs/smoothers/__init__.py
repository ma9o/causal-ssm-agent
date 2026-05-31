"""MPGibbs latent smoother registry."""

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    _LATENT_SMOOTHER_AMALA,
    _LATENT_SMOOTHER_AMALA_PLUS,
    _LATENT_SMOOTHER_DSMC,
    _LATENT_SMOOTHER_MGRAD,
    _LATENT_SMOOTHER_PLAIN,
)

from . import amala, amala_plus, dsmc, mgrad, plain_csmc

SMOOTHERS = {
    _LATENT_SMOOTHER_PLAIN: plain_csmc.smooth,
    _LATENT_SMOOTHER_AMALA: amala.smooth,
    _LATENT_SMOOTHER_AMALA_PLUS: amala_plus.smooth,
    _LATENT_SMOOTHER_MGRAD: mgrad.smooth,
    _LATENT_SMOOTHER_DSMC: dsmc.smooth,
}
