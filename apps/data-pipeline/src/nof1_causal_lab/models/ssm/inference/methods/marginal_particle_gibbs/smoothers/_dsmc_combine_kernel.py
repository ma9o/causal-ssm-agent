"""Pallas flash-style combine kernel for the dSMC seam evidence.

For one tree seam this computes

    log_joint[m, n] = logsumexp_p( a_p[m] + b_p[n] + <wm_p[m], wn_p[n]> )

which is algebraically the dense combine
``logsumexp_p(logpi_p + left_psi[m,p] + right_psi[n,p] + transition_lp[m,n,p])``
with the whitened transition vectors ``wm/wn`` and the per-(p, row) folds ``a/b``
precomputed in JAX (O(N*D*P), cheap). The reduction over the P parameter particles
is an *online* logsumexp (the flash-attention running-max rescale), so the
``(P, N, N)`` tensor never reaches HBM — only the ``(N, N)`` result is written. That
materialization is what made the dense combine bandwidth-bound; eliminating it is a
~45x cut in combine HBM traffic (measured on L4).

PRECISION — the cross is ``pl.dot``. The per-dimension multiply-add alternative
(``wm_p[:, d]``) does *not* lower on the Triton GPU backend ("Unimplemented
primitive: slice"), and Mosaic (the default GPU backend) can't lower an in-kernel
``dot`` at all — so ``pl.dot`` via Triton is the only cross that lowers. Measured on
an L4 (jax 0.9.0.1) it agrees with the dense fp32 reference to ``max|Δ| ~ 1e-6`` (the
fp32 ULP): in this build the dot is effectively fp32, not the tf32 we had budgeted
for. Robustness note: even if a Triton version ran the dot in tf32 (~5e-3 error), we
benchmarked the end-to-end effect (paired draws from a live conditioning state,
M=512, common random numbers) and found **no detectable bias** — in neither the
sampled latent paths nor the path complete-log-posterior — because the combine weight
is a *proposal/resampling* weight inside a reference-preserving cSMC-PGibbs kernel,
not a target potential, so it only reshuffles which near-tied pair is selected and
that averages out. On CPU the kernel runs in Pallas interpret mode (plain fp32,
exact; used by the test suite).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def _combine_kernel(wm_ref, wn_ref, a_ref, b_ref, out_ref, *, num_params):
    dtype = wm_ref.dtype
    bm = wm_ref.shape[1]
    bn = wn_ref.shape[1]
    acc_max = jnp.full((bm, bn), -jnp.inf, dtype=dtype)
    acc_sum = jnp.zeros((bm, bn), dtype=dtype)
    for p in range(num_params):
        wm_p = wm_ref[p]
        wn_p = wn_ref[p]
        cross = pl.dot(wm_p, wn_p, trans_b=True)  # only cross that lowers on Triton (see docstring)
        term = a_ref[p][:, None] + b_ref[p][None, :] + cross
        new_max = jnp.maximum(acc_max, term)
        acc_sum = acc_sum * jnp.exp(acc_max - new_max) + jnp.exp(term - new_max)
        acc_max = new_max
    out_ref[...] = acc_max + jnp.log(acc_sum)


def combine_log_joint(wm, wn, a, b, *, block=128, interpret=False):
    """``(N, N)`` seam log-joint via the streaming P-reduction kernel (no ``(P,N,N)``).

    ``wm, wn`` are ``(P, N, D)`` whitened vectors; ``a, b`` are ``(P, N)`` per-(p, row)
    folds. ``interpret=True`` runs the exact-fp32 CPU emulation (tests); otherwise it
    lowers to Triton on GPU.
    """
    num_params, n, latent_dim = wm.shape
    # tl.dot needs the contraction (latent) dim a multiple of 16; pad with zeros
    # (padded dims add 0 to the cross). Also satisfies the 16-byte row-load alignment.
    pad = (-latent_dim) % 16
    if pad:
        wm = jnp.pad(wm, ((0, 0), (0, 0), (0, pad)))
        wn = jnp.pad(wn, ((0, 0), (0, 0), (0, pad)))
        latent_dim += pad
    bm = bn = min(block, n)
    return pl.pallas_call(
        functools.partial(_combine_kernel, num_params=num_params),
        grid=(n // bm, n // bn),
        in_specs=[
            pl.BlockSpec(
                block_shape=(num_params, bm, latent_dim), index_map=lambda i, _j: (0, i, 0)
            ),
            pl.BlockSpec(
                block_shape=(num_params, bn, latent_dim), index_map=lambda _i, j: (0, j, 0)
            ),
            pl.BlockSpec(block_shape=(num_params, bm), index_map=lambda i, _j: (0, i)),
            pl.BlockSpec(block_shape=(num_params, bn), index_map=lambda _i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec(block_shape=(bm, bn), index_map=lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((n, n), wm.dtype),
        interpret=interpret,
        # Triton is the only backend that lowers an in-kernel dot on GPU (Ampere+/L4);
        # its tf32 is benchmarked unbiased here (see module docstring). CPU -> interpret.
        backend=None if interpret else "triton",
    )(wm, wn, a, b)
