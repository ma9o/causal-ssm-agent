"""First-pass Rao-Blackwellization: graph analysis for model-level decomposition.

Analyzes the SSMSpec structure (drift sparsity, observation dependencies, noise
families) to identify fully-decoupled linear-Gaussian sub-blocks that can be
marginalized exactly via Kalman filter before the particle filter runs.

This is the "first pass" — it operates on the model specification (fixed at
construction time), not on per-iteration parameter values. The resulting
partition is used by ComposedLikelihood to split the model into a Kalman
sub-model and a particle filter sub-model.

The "second pass" (block_rb.py) operates within each particle, marginalizing
Gaussian variables conditioned on sampled variables. Both passes compose:
first-pass removes unconditionally independent Gaussian blocks, second-pass
handles conditionally Gaussian blocks that couple to non-Gaussian variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from causal_ssm_agent.models.ssm.model import SSMSpec
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily, LinkFunction


@dataclass
class RBPartition:
    """Result of first-pass Rao-Blackwellization analysis.

    Indices into the original latent/observation variable ordering that
    define which variables go to the Kalman filter vs particle filter.
    """

    kalman_idx: np.ndarray  # latent var indices for Kalman
    particle_idx: np.ndarray  # latent var indices for PF
    obs_kalman_idx: np.ndarray  # obs channel indices for Kalman
    obs_particle_idx: np.ndarray  # obs channel indices for PF

    @property
    def has_kalman_block(self) -> bool:
        return len(self.kalman_idx) > 0

    @property
    def has_particle_block(self) -> bool:
        return len(self.particle_idx) > 0


def get_per_variable_diffusion(spec: SSMSpec) -> list[DistributionFamily]:
    """Resolve per-variable diffusion noise families.

    If spec.diffusion_dists is set, return it directly.
    Otherwise broadcast spec.diffusion_dist to all latent variables.
    """
    if spec.diffusion_dists is not None:
        return list(spec.diffusion_dists)
    return [spec.diffusion_dist] * spec.n_latent


def get_per_channel_links(spec: SSMSpec) -> list[LinkFunction]:
    """Resolve per-channel link functions.

    If spec.manifest_links is set, return it directly.
    Otherwise broadcast spec.manifest_link to all manifest channels.
    """
    if spec.manifest_links is not None:
        return list(spec.manifest_links)
    return [spec.manifest_link] * spec.n_manifest


def get_per_channel_manifest(spec: SSMSpec) -> list[DistributionFamily]:
    """Resolve per-channel observation noise families.

    If spec.manifest_dists is set, return it directly.
    Otherwise broadcast spec.manifest_dist to all manifest channels.
    """
    if spec.manifest_dists is not None:
        return list(spec.manifest_dists)
    return [spec.manifest_dist] * spec.n_manifest


def compute_drift_sparsity(spec: SSMSpec) -> np.ndarray:
    """Compute (n, n) boolean mask of potential nonzero drift entries.

    - drift_mask set -> use it directly (DAG-constrained)
    - drift="free", no mask -> all True (any entry could be nonzero)
    - fixed array -> True where abs(value) > 0
    """
    if spec.drift_mask is not None:
        return np.asarray(spec.drift_mask)
    n = spec.n_latent
    if isinstance(spec.drift, str):
        # "free" — all entries could be nonzero
        return np.ones((n, n), dtype=bool)
    arr = np.array(spec.drift)
    return np.abs(arr) > 0


def compute_obs_dependency(spec: SSMSpec) -> np.ndarray:
    """Compute (m, n) boolean mask of observation-to-latent dependencies.

    When lambda_mask is set, combines fixed nonzeros from lambda_mat
    with free positions from the mask.

    - lambda_mat="free" -> all True (any obs could depend on any latent)
    - fixed array + lambda_mask -> fixed nonzero | mask
    - fixed array, no mask -> True where abs(value) > 0
    """
    m, n = spec.n_manifest, spec.n_latent
    if isinstance(spec.lambda_mat, str):
        # "free" — all entries could be nonzero
        return np.ones((m, n), dtype=bool)
    arr = np.array(spec.lambda_mat)
    fixed_nonzero = np.abs(arr) > 0
    if spec.lambda_mask is not None:
        return fixed_nonzero | np.asarray(spec.lambda_mask)
    return fixed_nonzero


def analyze_first_pass_rb(spec: SSMSpec) -> RBPartition:
    """Graph-based first-pass Rao-Blackwellization via connected components.

    Identifies latent variables that form a fully-decoupled linear-Gaussian
    subsystem: no drift cross-coupling with non-Gaussian variables, no shared
    observations, and Gaussian noise on both diffusion and observation sides.

    Uses NetworkX connected_components on the drift coupling graph to find
    decoupled blocks, replacing a hand-rolled fixed-point iteration.

    Returns an RBPartition with index arrays for Kalman and particle blocks.
    """
    import networkx as nx

    n = spec.n_latent

    drift_mask = compute_drift_sparsity(spec)
    obs_dep = compute_obs_dependency(spec)
    per_var = get_per_variable_diffusion(spec)
    per_obs = get_per_channel_manifest(spec)

    # Step 1: Identify Gaussian-eligible variables (Gaussian diffusion +
    # all observation channels that depend on them have Gaussian obs noise)
    gaussian_eligible = set()
    for i in range(n):
        if per_var[i] != "gaussian":
            continue
        # Check that all obs channels depending on i have Gaussian obs noise
        has_nongaussian_obs = False
        for k in range(spec.n_manifest):
            if obs_dep[k, i] and per_obs[k] != "gaussian":
                has_nongaussian_obs = True
                break
        if not has_nongaussian_obs:
            gaussian_eligible.add(i)

    non_gaussian = set(range(n)) - gaussian_eligible

    # Step 2: Build undirected coupling graph from drift sparsity.
    # Edge (i, j) exists if drift_mask[i, j] or drift_mask[j, i] (any direction).
    # Then find connected components — a Gaussian-eligible variable that shares
    # a component with any non-Gaussian variable cannot be Kalman-marginalized.
    coupling_graph = nx.Graph()
    coupling_graph.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if drift_mask[i, j] or drift_mask[j, i]:
                coupling_graph.add_edge(i, j)

    # A connected component is Kalman-eligible only if it contains
    # no non-Gaussian variables.
    candidates = set()
    for component in nx.connected_components(coupling_graph):
        if component.isdisjoint(non_gaussian):
            candidates |= component

    # Step 3: Assign observation channels.
    # An obs channel goes to Kalman only if it depends exclusively on Kalman variables.
    obs_kalman = []
    obs_particle = []
    for k in range(spec.n_manifest):
        deps = set(np.where(obs_dep[k, :])[0])
        if deps and deps.issubset(candidates):
            obs_kalman.append(k)
        else:
            obs_particle.append(k)

    # Build partition
    kalman_idx = np.array(sorted(candidates), dtype=np.int32)
    particle_idx = np.array(sorted(set(range(n)) - candidates), dtype=np.int32)
    obs_kalman_idx = np.array(obs_kalman, dtype=np.int32)
    obs_particle_idx = np.array(obs_particle, dtype=np.int32)

    return RBPartition(
        kalman_idx=kalman_idx,
        particle_idx=particle_idx,
        obs_kalman_idx=obs_kalman_idx,
        obs_particle_idx=obs_particle_idx,
    )


def kalman_block_profile_indices(spec: SSMSpec, partition: RBPartition) -> list[int]:
    """Return flat parameter vector indices that belong to the Kalman block.

    Only these indices should be profiled in the parametric identifiability
    check — particle-block parameters have stochastic likelihoods that make
    profile curves unreliable.

    Mirrors the NumPyro site layout from SSMModel._sample_* methods exactly.
    """

    kalman_set = {int(i) for i in partition.kalman_idx}
    obs_kalman_set = {int(i) for i in partition.obs_kalman_idx}
    n = spec.n_latent
    m = spec.n_manifest
    indices: list[int] = []
    offset = 0

    # --- drift_diag_pop: shape (n,), index k → latent k ---
    if isinstance(spec.drift, str) and spec.drift == "free":
        for k in range(n):
            if k in kalman_set:
                indices.append(offset + k)
        offset += n

        # --- drift_offdiag_pop: shape (n_offdiag,) ---
        # Reconstruct offdiag_positions exactly as SSMModel._sample_drift does
        offdiag_positions: list[tuple[int, int]] = []
        if spec.drift_mask is not None:
            for i in range(n):
                for j in range(n):
                    if i != j and spec.drift_mask[i, j]:
                        offdiag_positions.append((i, j))
        else:
            for i in range(n):
                for j in range(n):
                    if i != j:
                        offdiag_positions.append((i, j))

        for idx, (i, j) in enumerate(offdiag_positions):
            if i in kalman_set and j in kalman_set:
                indices.append(offset + idx)
        offset += len(offdiag_positions)

    # --- diffusion_diag_pop: shape (n,), index k → latent k ---
    if isinstance(spec.diffusion, str):
        for k in range(n):
            if k in kalman_set:
                indices.append(offset + k)
        offset += n

        # --- diffusion_lower: shape (n*(n-1)//2,) ---
        if spec.diffusion == "free":
            n_lower = n * (n - 1) // 2
            lower_idx = 0
            for i in range(n):
                for j in range(i):
                    if i in kalman_set and j in kalman_set:
                        indices.append(offset + lower_idx)
                    lower_idx += 1
            offset += n_lower

    # --- cint_pop: shape (n,), index k → latent k ---
    if spec.cint is not None and isinstance(spec.cint, str) and spec.cint == "free":
        for k in range(n):
            if k in kalman_set:
                indices.append(offset + k)
        offset += n

    # --- lambda_free: variable layout depending on mode ---
    if isinstance(spec.lambda_mat, str) and spec.lambda_mat == "free":
        # Legacy mode: extra rows beyond identity
        if m > n:
            n_free = (m - n) * n
            idx = 0
            for i in range(n, m):
                for j in range(n):
                    if i in obs_kalman_set and j in kalman_set:
                        indices.append(offset + idx)
                    idx += 1
            offset += n_free
    elif not isinstance(spec.lambda_mat, str) and spec.lambda_mask is not None:
        # Template+mask mode
        for i in range(m):
            for j in range(n):
                if spec.lambda_mask[i, j]:
                    if i in obs_kalman_set and j in kalman_set:
                        indices.append(offset)
                    offset += 1

    # --- manifest_means: shape (m,), index k → manifest k ---
    if isinstance(getattr(spec, "manifest_means", None), str) and spec.manifest_means == "free":
        for k in range(m):
            if k in obs_kalman_set:
                indices.append(offset + k)
        offset += m

    # --- manifest_var_diag: shape (m,), index k → manifest k ---
    if isinstance(spec.manifest_var, str):
        for k in range(m):
            if k in obs_kalman_set:
                indices.append(offset + k)
        offset += m

    # --- t0_means_pop: shape (n,), index k → latent k ---
    if isinstance(spec.t0_means, str) and spec.t0_means == "free":
        for k in range(n):
            if k in kalman_set:
                indices.append(offset + k)
        offset += n

    # --- t0_var_diag: shape (n,), index k → latent k ---
    if isinstance(spec.t0_var, str):
        for k in range(n):
            if k in kalman_set:
                indices.append(offset + k)
        offset += n

    # Noise family hyperparams (obs_df, proc_df, etc.) are global scalars —
    # include only if the entire model is Kalman-tractable.
    if not partition.has_particle_block:
        # All remaining scalar sites are Kalman-safe
        # (obs_df, obs_shape, obs_r, obs_concentration, proc_df)
        # They are appended after the above sites in alphabetical order
        # by _discover_sites. We don't know exactly how many there are
        # without tracing, so we skip them in the mixed case.
        pass

    return indices
