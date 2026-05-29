"""MPGibbs latent smoother registry."""

from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs._contract import (
    _LATENT_SMOOTHER_AMALA,
    _LATENT_SMOOTHER_MGRAD,
    _LATENT_SMOOTHER_PLAIN,
)
from nof1_causal_lab.models.ssm.inference.methods.marginal_particle_gibbs.smoothers import (
    amala,
    mgrad,
    plain_csmc,
)

SMOOTHERS = {
    _LATENT_SMOOTHER_PLAIN: plain_csmc.smooth,
    _LATENT_SMOOTHER_AMALA: amala.smooth,
    _LATENT_SMOOTHER_MGRAD: mgrad.smooth,
}
