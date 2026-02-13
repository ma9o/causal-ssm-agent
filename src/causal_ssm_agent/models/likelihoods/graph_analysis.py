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


def get_per_variable_diffusion(spec: SSMSpec) -> list[str]:
    """Resolve per-variable diffusion noise families.

    If spec.diffusion_dists is set, return it as string list.
    Otherwise broadcast spec.diffusion_dist to all latent variables.
    """
    if spec.diffusion_dists is not None:
        return [d.value for d in spec.diffusion_dists]
    return [spec.diffusion_dist.value] * spec.n_latent


def get_per_channel_manifest(spec: SSMSpec) -> list[str]:
    """Resolve per-channel observation noise families.

    If spec.manifest_dists is set, return it as string list.
    Otherwise broadcast spec.manifest_dist to all manifest channels.
    """
    if spec.manifest_dists is not None:
        return [d.value for d in spec.manifest_dists]
    return [spec.manifest_dist.value] * spec.n_manifest


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
