"""Standalone Pallas flash-style combine kernel for the dSMC seam evidence.

Isolates the one op that dominates dsmc bandwidth (the `(P,N,N)` materialization
in `_stitch_logits`):

    log_joint[m,n] = logsumexp_p( base_p + a_p[m] + b_p[n] + <wm_p[m], wn_p[n]> )

where the per-particle whitened vectors `wm,wn` (P,N,D) and the per-(p,row) folds
`a,b` (P,N) and per-param constant `base` (P,) are precomputed cheaply in JAX
(O(N·D·P)). The kernel reduces over P with an online logsumexp (flash-attention
rescale) so the `(P,N,N)` tensor never touches HBM — output is `(N,N)`.

Test correctness locally with no GPU:   uv run python scripts/pallas_combine.py --local
Measure bytes/perf on an L4:            modal run scripts/pallas_combine.py
"""

from __future__ import annotations

import argparse
import functools
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import modal
from jax.experimental import pallas as pl

_LOG_2PI = math.log(2.0 * math.pi)


def _combine_kernel(wm_ref, wn_ref, a_ref, b_ref, out_ref, *, num_params):
    bm = wm_ref.shape[1]
    bn = wn_ref.shape[1]
    acc_max = jnp.full((bm, bn), -jnp.inf, dtype=jnp.float32)
    acc_sum = jnp.zeros((bm, bn), dtype=jnp.float32)
    for p in range(num_params):
        # cross[m,n] = <wm_p[m], wn_p[n]> via tl.dot. The per-dimension multiply-add
        # alternative (wm_p[:, d]) does NOT lower on Triton ("Unimplemented primitive:
        # slice"). tl.dot runs in tf32 regardless of precision/allow_tf32 (~5e-3 error);
        # benchmarked unbiased end-to-end (see _dsmc_combine_kernel docstring).
        wm_p = wm_ref[p]
        wn_p = wn_ref[p]
        cross = pl.dot(wm_p, wn_p, trans_b=True)
        term = a_ref[p][:, None] + b_ref[p][None, :] + cross
        new_max = jnp.maximum(acc_max, term)
        acc_sum = acc_sum * jnp.exp(acc_max - new_max) + jnp.exp(term - new_max)
        acc_max = new_max
    out_ref[...] = acc_max + jnp.log(acc_sum)


def log_joint_pallas(wm, wn, a, b, base, *, block_m=128, block_n=128, interpret=False):
    """(N,N) seam log-joint via the flash-style P-reduction kernel."""
    num_params, n, latent_dim = wm.shape
    # tl.dot needs the contraction (latent) dim a multiple of 16; pad with zeros
    # (padded dims add 0 to the cross). Also satisfies the 16-byte row-load alignment.
    pad = (-latent_dim) % 16
    if pad:
        wm = jnp.pad(wm, ((0, 0), (0, 0), (0, pad)))
        wn = jnp.pad(wn, ((0, 0), (0, 0), (0, pad)))
        latent_dim += pad
    # Fold the per-param constant into `a`: avoids a tiny (P,)=32B kernel input, since
    # Mosaic requires every copy to be a multiple of the 128B warpgroup size.
    a = a + base[:, None]
    grid = (n // block_m, n // block_n)
    return pl.pallas_call(
        functools.partial(_combine_kernel, num_params=num_params),
        grid=grid,
        in_specs=[
            pl.BlockSpec(
                block_shape=(num_params, block_m, latent_dim), index_map=lambda i, j: (0, i, 0)
            ),
            pl.BlockSpec(
                block_shape=(num_params, block_n, latent_dim), index_map=lambda i, j: (0, j, 0)
            ),
            pl.BlockSpec(block_shape=(num_params, block_m), index_map=lambda i, j: (0, i)),
            pl.BlockSpec(block_shape=(num_params, block_n), index_map=lambda i, j: (0, j)),
        ],
        out_specs=pl.BlockSpec(block_shape=(block_m, block_n), index_map=lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((n, n), jnp.float32),
        interpret=interpret,
        # Mosaic (default) can't lower in-kernel dot_general; Triton (Ampere+, incl L4) can.
        backend=None if interpret else "triton",
    )(wm, wn, a, b)


def log_joint_reference(wm, wn, a, b, base):
    """Dense JAX reference — materializes the (P,N,N) tensor the kernel avoids."""
    cross = jnp.einsum("pmd,pnd->pmn", wm, wn)
    term = base[:, None, None] + a[:, :, None] + b[:, None, :] + cross
    return jax.scipy.special.logsumexp(term, axis=0)


def _random_inputs(num_particles, num_params, latent_dim, *, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    wm = jax.random.normal(keys[0], (num_params, num_particles, latent_dim), jnp.float32)
    wn = jax.random.normal(keys[1], (num_params, num_particles, latent_dim), jnp.float32)
    a = jax.random.normal(keys[2], (num_params, num_particles), jnp.float32)
    b = jax.random.normal(keys[3], (num_params, num_particles), jnp.float32)
    base = jax.random.normal(keys[4], (num_params,), jnp.float32)
    return wm, wn, a, b, base


def run_check(*, num_particles=512, num_params=8, latent_dim=3, block=128, interpret=False):
    wm, wn, a, b, base = _random_inputs(num_particles, num_params, latent_dim)
    ref = jax.jit(log_joint_reference)
    pallas = jax.jit(
        functools.partial(log_joint_pallas, block_m=block, block_n=block, interpret=interpret)
    )
    out_ref = ref(wm, wn, a, b, base)
    out_pal = pallas(wm, wn, a, b, base)
    out_ref.block_until_ready()
    out_pal.block_until_ready()
    max_abs = float(jnp.max(jnp.abs(out_ref - out_pal)))
    result = {
        "N": num_particles,
        "P": num_params,
        "D": latent_dim,
        "block": block,
        "interpret": interpret,
        "max_abs_diff": max_abs,
        "allclose": bool(jnp.allclose(out_ref, out_pal, atol=1e-4, rtol=1e-4)),
    }
    if not interpret:
        rc = ref.lower(wm, wn, a, b, base).compile().cost_analysis()
        pc = pallas.lower(wm, wn, a, b, base).compile().cost_analysis()

        def _bytes(c):
            return (
                float(c["bytes accessed"]) if isinstance(c, dict) else float(c[0]["bytes accessed"])
            )

        result["ref_bytes"] = _bytes(rc)
        result["pallas_bytes"] = _bytes(pc)
        result["bytes_ratio"] = result["ref_bytes"] / max(result["pallas_bytes"], 1.0)
    return result


image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # Triton Pallas backend (Ampere+, runs on L4). Pin jax==0.9.0.1 so the unpinned
    # triton resolve can't drag jax backwards (which dropped the `backend` kwarg before).
    .uv_pip_install("jax[cuda12]==0.9.0.1", "triton", "absl-py", gpu="L4")
)
app = modal.App("nof1-dsmc-pallas-combine", image=image)


@app.function(gpu="L4", timeout=900)
def gpu_check() -> list[dict]:
    import jax as _jax

    rows = []
    print("JAX devices:", _jax.devices(), flush=True)
    for n in (128, 256, 512):
        block = min(128, n)
        rows.append(run_check(num_particles=n, block=block, interpret=False))
        print(rows[-1], flush=True)
    return rows


@app.local_entrypoint()
def main():
    for row in gpu_check.remote():
        ratio = row.get("bytes_ratio")
        print(
            f"N={row['N']:>4} P={row['P']} D={row['D']} block={row['block']} | "
            f"allclose={row['allclose']} max|Δ|={row['max_abs_diff']:.2e} | "
            f"ref={row['ref_bytes'] / 1e6:.1f}MB pallas={row['pallas_bytes'] / 1e6:.1f}MB "
            f"({ratio:.1f}x less)"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="CPU interpret-mode correctness check")
    args = parser.parse_args()
    if args.local:
        print(run_check(num_particles=256, block=64, interpret=True))
    else:
        print(f"Run on GPU with: modal run {Path(__file__).name}")
