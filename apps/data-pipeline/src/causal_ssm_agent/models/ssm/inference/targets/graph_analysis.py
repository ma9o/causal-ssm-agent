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

from causal_ssm_agent.models.ssm.assembler import SSMAssembler

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
        """True when any component (latent or observation) is non-Kalman."""
        return len(self.particle_idx) > 0 or len(self.obs_particle_idx) > 0


def get_per_variable_diffusion(spec: SSMSpec) -> list[DistributionFamily]:
    """Return the canonical per-variable diffusion noise families."""
    return list(spec.diffusion_dists)


def has_student_t_diffusion(spec: SSMSpec) -> bool:
    """Return whether any latent process uses Student-t diffusion noise."""
    from causal_ssm_agent.orchestrator.schemas_model import DistributionFamily

    return DistributionFamily.STUDENT_T in set(get_per_variable_diffusion(spec))


def get_per_channel_links(spec: SSMSpec) -> list[LinkFunction]:
    """Resolve per-channel link functions.

    If spec.manifest_links is set, return it directly.
    Otherwise use the default link for each channel's observation family.
    """
    from causal_ssm_agent.models.ssm.inference.targets.observation_families import (
        resolve_manifest_families_and_links,
    )

    if spec.manifest_links is not None:
        return list(spec.manifest_links)
    _, links = resolve_manifest_families_and_links(
        get_per_channel_manifest(spec),
    )
    return links


def get_per_channel_manifest(spec: SSMSpec) -> list[DistributionFamily]:
    """Return the canonical per-channel observation noise families."""
    return list(spec.manifest_dists)


def compute_drift_sparsity(spec: SSMSpec) -> np.ndarray:
    """Compute (n, n) boolean mask of potential nonzero drift entries.

    Potential nonzeros are the union of fixed nonzero template entries and
    free entries marked in the compiled drift mask.
    """
    arr = np.array(spec.drift)
    fixed_nonzero = np.abs(arr) > 0
    drift_free = np.asarray(spec.drift_offdiag_mask).copy()
    np.fill_diagonal(drift_free, np.asarray(spec.drift_diag_mask))
    return fixed_nonzero | drift_free


def compute_obs_dependency(spec: SSMSpec) -> np.ndarray:
    """Compute (m, n) boolean mask of observation-to-latent dependencies.

    Combines the fixed loading template with the explicit free-loading mask.
    """
    arr = np.array(spec.lambda_mat)
    fixed_nonzero = np.abs(arr) > 0
    return fixed_nonzero | np.asarray(spec.lambda_mask)


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
    per_links = get_per_channel_links(spec)

    # Step 1: Identify Gaussian-eligible variables (Gaussian diffusion +
    # all observation channels that depend on them have Gaussian obs noise
    # AND identity link function — non-identity links make the observation
    # model nonlinear, which the Kalman filter cannot handle)
    gaussian_eligible = set()
    for i in range(n):
        if per_var[i] != "gaussian":
            continue
        # Check that all obs channels depending on i have Gaussian obs noise
        # and identity link function
        has_nonlinear_obs = False
        for k in range(spec.n_manifest):
            if obs_dep[k, i] and (per_obs[k] != "gaussian" or per_links[k] != "identity"):
                has_nonlinear_obs = True
                break
        if not has_nonlinear_obs:
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
    # An obs channel goes to Kalman only if it depends exclusively on Kalman
    # variables. Zero-dependency channels (empty loading row) are Kalman-safe
    # only when their own noise family is Gaussian with identity link;
    # otherwise they need the particle/non-Kalman observation bucket.
    obs_kalman = []
    obs_particle = []
    for k in range(spec.n_manifest):
        deps = set(np.where(obs_dep[k, :])[0])
        if deps:
            if deps.issubset(candidates):
                obs_kalman.append(k)
            else:
                obs_particle.append(k)
        else:
            # Zero-dependency channel: Kalman-compatible only if Gaussian+identity
            if per_obs[k] == "gaussian" and per_links[k] == "identity":
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

    assembler = SSMAssembler(spec)
    kalman_set = {int(i) for i in partition.kalman_idx}
    obs_kalman_set = {int(i) for i in partition.obs_kalman_idx}
    n = spec.n_latent
    m = spec.n_manifest
    indices: list[int] = []
    offset = 0

    # --- drift_diag_pop: shape (n,), index k → latent k ---
    for dense_idx, latent_idx in enumerate(assembler.drift_diag_positions):
        if latent_idx in kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.drift_diag_positions)

    for idx, (i, j) in enumerate(assembler.offdiag_positions):
        if i in kalman_set and j in kalman_set:
            indices.append(offset + idx)
    offset += len(assembler.offdiag_positions)

    for dense_idx, latent_idx in enumerate(assembler.diffusion_diag_positions):
        if latent_idx in kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.diffusion_diag_positions)

    for dense_idx, (row, col) in enumerate(assembler.diffusion_lower_positions):
        if row in kalman_set and col in kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.diffusion_lower_positions)

    # --- cint_pop: sparse free continuous intercept entries ---
    for dense_idx, latent_idx in enumerate(assembler.cint_free_positions):
        if latent_idx in kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.cint_free_positions)

    # --- lambda_free: iterate the canonical free-loading mask ---
    for i in range(m):
        for j in range(n):
            if spec.lambda_mask[i, j]:
                if i in obs_kalman_set and j in kalman_set:
                    indices.append(offset)
                offset += 1

    # --- manifest_means: sparse free manifest intercept entries ---
    for dense_idx, manifest_idx in enumerate(assembler.manifest_means_free_positions):
        if manifest_idx in obs_kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.manifest_means_free_positions)

    for dense_idx, manifest_idx in enumerate(assembler.manifest_var_free_positions):
        if manifest_idx in obs_kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.manifest_var_free_positions)

    # --- t0_means_pop: sparse free initial-state mean entries ---
    for dense_idx, latent_idx in enumerate(assembler.t0_means_free_positions):
        if latent_idx in kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.t0_means_free_positions)

    for dense_idx, latent_idx in enumerate(assembler.t0_diag_free_positions):
        if latent_idx in kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.t0_diag_free_positions)

    for dense_idx, (row, col) in enumerate(assembler.t0_correlation_positions):
        if row in kalman_set and col in kalman_set:
            indices.append(offset + dense_idx)
    offset += len(assembler.t0_correlation_positions)

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
