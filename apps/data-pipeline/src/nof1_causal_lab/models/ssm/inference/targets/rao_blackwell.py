"""Rao-Blackwell partition artifacts for runtime-conditioned SSM inference.

RBPF modes are explicit because they expose different particle-kernel contracts:

* ``independent`` collapses linear-Gaussian marginalized blocks that can be scored
  without carrying a path-dependent Kalman filter state. PIT particle kernels keep
  their time-separable tree structure in this mode.
* ``conditional`` collapses marginalized blocks whose linear-Gaussian dynamics or
  observations depend on the carried state. Each particle prefix owns a Kalman
  filter state, so PIT particle kernels must switch to a sequential conditional
  RBPF update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, NamedTuple, cast

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction
from nof1_causal_lab.models.ssm.covariance_utils import symmetrize_with_jitter
from nof1_causal_lab.models.ssm.inference.targets.polya_gamma import (
    negative_binomial_finite_sum_base_log_terms,
    polya_gamma_gaussian_logpdf_correction,
)

RBPFMode = Literal["none", "independent", "conditional"]
SUPPORTED_RBPF_MODES: tuple[RBPFMode, ...] = ("none", "independent", "conditional")


@dataclass(frozen=True)
class RBPFPartitionSpec:
    """Static latent partition consumed by RBPF-conditioned runtimes."""

    carried_latent_indices: tuple[int, ...]
    marginalized_latent_indices: tuple[int, ...]

    @property
    def is_full_path(self) -> bool:
        return len(self.marginalized_latent_indices) == 0


@dataclass(frozen=True)
class RBPFPartitionDiagnostics:
    """Host-side explanation of an automatically derived RBPF partition."""

    mode: RBPFMode
    requested_marginalized_latent_indices: tuple[int, ...] | None
    candidate_marginalized_latent_indices: tuple[int, ...]
    carried_latent_indices: tuple[int, ...]
    marginalized_latent_indices: tuple[int, ...]
    forced_carried_latent_indices: tuple[int, ...]
    forced_carried: tuple[dict[str, Any], ...]
    gaussian_observation_channels: tuple[int, ...]
    polya_gamma_observation_channels: tuple[int, ...]
    residual_observation_channels: tuple[int, ...]

    def asdict(self) -> dict[str, Any]:
        """Return a JSON-serializable diagnostic payload."""
        return {
            "mode": self.mode,
            "requested_marginalized_latent_indices": None
            if self.requested_marginalized_latent_indices is None
            else list(self.requested_marginalized_latent_indices),
            "candidate_marginalized_latent_indices": list(
                self.candidate_marginalized_latent_indices
            ),
            "carried_latent_indices": list(self.carried_latent_indices),
            "marginalized_latent_indices": list(self.marginalized_latent_indices),
            "forced_carried_latent_indices": list(self.forced_carried_latent_indices),
            "forced_carried": [dict(item) for item in self.forced_carried],
            "gaussian_observation_channels": list(self.gaussian_observation_channels),
            "polya_gamma_observation_channels": list(self.polya_gamma_observation_channels),
            "residual_observation_channels": list(self.residual_observation_channels),
        }


@dataclass(frozen=True)
class RBPFObservationPlan:
    """Manifest rows consumed by a collapsed Gaussian RBPF block."""

    channel_mask: jnp.ndarray
    gaussian_channel_mask: jnp.ndarray
    polya_gamma_channel_mask: jnp.ndarray
    negative_binomial_polya_gamma_channel_mask: jnp.ndarray
    structure: str

    @property
    def enabled(self) -> bool:
        return bool(np.any(np.asarray(self.channel_mask)))


class RBPFMarginalContext(NamedTuple):
    """Linear-Gaussian subsystem integrated out conditional on sampled parameters."""

    Ad_cc: jnp.ndarray
    Ad_mm: jnp.ndarray
    Ad_mc: jnp.ndarray
    Q_cc: jnp.ndarray
    Q_mm: jnp.ndarray
    Q_mc: jnp.ndarray
    cd_c: jnp.ndarray
    cd_m: jnp.ndarray
    init_mean_c: jnp.ndarray
    init_mean_m: jnp.ndarray
    init_cov_cc: jnp.ndarray
    init_cov_mm: jnp.ndarray
    init_cov_mc: jnp.ndarray
    H_m: jnp.ndarray
    H_c: jnp.ndarray
    H_m_rows: jnp.ndarray | None
    H_c_rows: jnp.ndarray | None
    d_meas: jnp.ndarray
    d_meas_rows: jnp.ndarray | None
    R: jnp.ndarray
    observation_indices: jnp.ndarray
    gaussian_row_mask: jnp.ndarray
    polya_gamma_row_mask: jnp.ndarray
    negative_binomial_polya_gamma_row_mask: jnp.ndarray
    obs_r: jnp.ndarray


class RBPFMarginalFilterState(NamedTuple):
    """Filtered marginalized-state moments for one carried particle/path prefix."""

    mean: jnp.ndarray
    cov: jnp.ndarray


def normalize_rbpf_mode(rbpf_mode: str) -> RBPFMode:
    """Normalize and validate the public RBPF runtime-conditioning mode."""
    mode = str(rbpf_mode).strip().lower()
    if mode not in SUPPORTED_RBPF_MODES:
        raise ValueError(
            f"Unsupported rbpf_mode {rbpf_mode!r}. "
            f"Supported: {', '.join(repr(mode) for mode in SUPPORTED_RBPF_MODES)}."
        )
    return cast("RBPFMode", mode)


def full_path_rbpf_partition(n_latent: int) -> RBPFPartitionSpec:
    """Return the identity partition: every latent dimension remains sampled."""
    if n_latent < 1:
        raise ValueError(f"n_latent must be positive, got {n_latent}.")
    return RBPFPartitionSpec(
        carried_latent_indices=tuple(range(int(n_latent))),
        marginalized_latent_indices=(),
    )


def build_rbpf_partition(
    n_latent: int,
    marginalized_latent_indices: tuple[int, ...] | list[int] | None,
) -> RBPFPartitionSpec:
    """Build a static partition from marginalized latent indices."""
    if marginalized_latent_indices is None:
        return full_path_rbpf_partition(n_latent)
    marginalized = tuple(int(idx) for idx in marginalized_latent_indices)
    marginalized_set = set(marginalized)
    carried = tuple(idx for idx in range(int(n_latent)) if idx not in marginalized_set)
    partition = RBPFPartitionSpec(
        carried_latent_indices=carried,
        marginalized_latent_indices=marginalized,
    )
    validate_rbpf_partition(partition, n_latent=n_latent)
    return partition


def _validate_candidate_indices(
    n_latent: int,
    marginalized_latent_indices: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...]:
    if marginalized_latent_indices is None:
        return tuple(range(int(n_latent)))
    marginalized = tuple(int(idx) for idx in marginalized_latent_indices)
    if len(set(marginalized)) != len(marginalized):
        raise ValueError(
            f"RBPF candidate marginalized latent indices must be unique; got {marginalized}."
        )
    invalid = tuple(idx for idx in marginalized if idx < 0 or idx >= int(n_latent))
    if invalid:
        raise ValueError(
            "RBPF candidate marginalized latent indices must be in "
            f"[0, {int(n_latent)}); got {invalid}."
        )
    return tuple(sorted(marginalized))


def _latent_name(spec, idx: int) -> str:
    names = getattr(spec, "latent_names", None) or ()
    if idx < len(names):
        return str(names[idx])
    return f"latent_{idx}"


def _manifest_name(spec, idx: int) -> str:
    names = getattr(spec, "manifest_names", None) or ()
    if idx < len(names):
        return str(names[idx])
    return f"manifest_{idx}"


def _observation_channel_masks(
    spec,
    manifest_links: list[LinkFunction],
    polya_gamma_channel_mask: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pg_channels = np.asarray(polya_gamma_channel_mask, dtype=bool)
    if pg_channels.shape != (spec.n_manifest,):
        raise ValueError(
            "polya_gamma_channel_mask must have shape "
            f"({spec.n_manifest},), got {pg_channels.shape}."
        )
    gaussian_channels = np.zeros((spec.n_manifest,), dtype=bool)
    residual_channels = np.zeros((spec.n_manifest,), dtype=bool)
    for channel_idx, (dist, link) in enumerate(
        zip(spec.manifest_dists, manifest_links, strict=True)
    ):
        dist_value = _enum_value_lower(dist)
        link_value = _enum_value_lower(link)
        is_gaussian_identity = dist_value == _enum_value_lower(
            DistributionFamily.GAUSSIAN
        ) and link_value == _enum_value_lower(LinkFunction.IDENTITY)
        gaussian_channels[channel_idx] = is_gaussian_identity
        residual_channels[channel_idx] = not is_gaussian_identity and not pg_channels[channel_idx]
    return gaussian_channels, pg_channels, residual_channels


def _force_carried(
    *,
    carried: set[int],
    forced: list[dict[str, Any]],
    spec,
    latent_idx: int,
    reason: dict[str, Any],
) -> bool:
    if int(latent_idx) in carried:
        return False
    idx = int(latent_idx)
    carried.add(idx)
    forced.append(
        {
            "latent_index": idx,
            "latent_name": _latent_name(spec, idx),
            **reason,
        }
    )
    return True


def _force_residual_observation_latents(
    *,
    spec,
    manifest_links: list[LinkFunction],
    loading_active: np.ndarray,
    residual_channels: np.ndarray,
    carried: set[int],
    forced: list[dict[str, Any]],
) -> bool:
    changed = False
    for channel_idx in np.where(residual_channels)[0].tolist():
        dist_value = _enum_value_lower(spec.manifest_dists[channel_idx])
        link_value = _enum_value_lower(manifest_links[channel_idx])
        for latent_idx in np.where(loading_active[channel_idx])[0].tolist():
            changed = (
                _force_carried(
                    carried=carried,
                    forced=forced,
                    spec=spec,
                    latent_idx=int(latent_idx),
                    reason={
                        "reason": "residual_observation",
                        "observation_index": int(channel_idx),
                        "observation_name": _manifest_name(spec, int(channel_idx)),
                        "distribution": dist_value,
                        "link": link_value,
                    },
                )
                or changed
            )
    return changed


def _force_independent_mixed_observation_latents(
    *,
    spec,
    loading_active: np.ndarray,
    carried: set[int],
    forced: list[dict[str, Any]],
) -> bool:
    carried_indices = tuple(sorted(carried))
    marginalized_indices = tuple(idx for idx in range(spec.n_latent) if idx not in carried)
    if not carried_indices or not marginalized_indices:
        return False
    uses_carried = np.any(loading_active[:, carried_indices], axis=1)
    uses_marginalized = np.any(loading_active[:, marginalized_indices], axis=1)
    changed = False
    for channel_idx in np.where(uses_carried & uses_marginalized)[0].tolist():
        for latent_idx in np.where(loading_active[channel_idx])[0].tolist():
            if int(latent_idx) not in carried:
                changed = (
                    _force_carried(
                        carried=carried,
                        forced=forced,
                        spec=spec,
                        latent_idx=int(latent_idx),
                        reason={
                            "reason": "independent_mode_mixed_observation",
                            "observation_index": int(channel_idx),
                            "observation_name": _manifest_name(spec, int(channel_idx)),
                        },
                    )
                    or changed
                )
    return changed


def _force_cross_covariance_latents(
    *,
    spec,
    active: np.ndarray,
    carried: set[int],
    forced: list[dict[str, Any]],
    reason: str,
) -> bool:
    changed = False
    carried_indices = tuple(sorted(carried))
    if not carried_indices:
        return False
    for latent_idx in range(spec.n_latent):
        if latent_idx in carried:
            continue
        connected = tuple(
            carried_idx
            for carried_idx in carried_indices
            if bool(active[latent_idx, carried_idx]) or bool(active[carried_idx, latent_idx])
        )
        if connected:
            changed = (
                _force_carried(
                    carried=carried,
                    forced=forced,
                    spec=spec,
                    latent_idx=latent_idx,
                    reason={
                        "reason": reason,
                        "connected_carried_latent_indices": list(connected),
                    },
                )
                or changed
            )
    return changed


def _force_dynamics_closure(
    *,
    spec,
    rbpf_mode: RBPFMode,
    carried: set[int],
    forced: list[dict[str, Any]],
) -> bool:
    changed = False
    for component_index, component in enumerate(spec.dynamics_spec.components):
        component_name = type(component).__name__
        if hasattr(component, "source") and hasattr(component, "target"):
            source = int(component.source)
            target = int(component.target)
            source_carried = source in carried
            target_carried = target in carried
            is_linear = component_name == "LinearEdgeSpec"
            if target_carried and not source_carried:
                changed = (
                    _force_carried(
                        carried=carried,
                        forced=forced,
                        spec=spec,
                        latent_idx=source,
                        reason={
                            "reason": "dynamics_to_carried",
                            "component_index": int(component_index),
                            "component": component_name,
                            "source": source,
                            "target": target,
                        },
                    )
                    or changed
                )
                continue
            if not target_carried and not is_linear:
                changed = (
                    _force_carried(
                        carried=carried,
                        forced=forced,
                        spec=spec,
                        latent_idx=target,
                        reason={
                            "reason": "nonlinear_marginalized_dynamics",
                            "component_index": int(component_index),
                            "component": component_name,
                            "source": source,
                            "target": target,
                        },
                    )
                    or changed
                )
                continue
            if rbpf_mode == "independent" and source_carried and not target_carried:
                changed = (
                    _force_carried(
                        carried=carried,
                        forced=forced,
                        spec=spec,
                        latent_idx=target,
                        reason={
                            "reason": "independent_mode_dynamics_dependency",
                            "component_index": int(component_index),
                            "component": component_name,
                            "source": source,
                            "target": target,
                        },
                    )
                    or changed
                )
        elif (
            hasattr(component, "source_a")
            and hasattr(component, "source_b")
            and hasattr(component, "target")
        ):
            source_a = int(component.source_a)
            source_b = int(component.source_b)
            target = int(component.target)
            if target in carried:
                for source in (source_a, source_b):
                    if source not in carried:
                        changed = (
                            _force_carried(
                                carried=carried,
                                forced=forced,
                                spec=spec,
                                latent_idx=source,
                                reason={
                                    "reason": "dynamics_to_carried",
                                    "component_index": int(component_index),
                                    "component": component_name,
                                    "source": source,
                                    "target": target,
                                },
                            )
                            or changed
                        )
            elif target not in carried:
                changed = (
                    _force_carried(
                        carried=carried,
                        forced=forced,
                        spec=spec,
                        latent_idx=target,
                        reason={
                            "reason": "nonlinear_marginalized_dynamics",
                            "component_index": int(component_index),
                            "component": component_name,
                            "source_a": source_a,
                            "source_b": source_b,
                            "target": target,
                        },
                    )
                    or changed
                )
    return changed


def _derive_rbpf_partition_from_carried(
    *,
    spec,
    rbpf_mode: RBPFMode,
    candidate_marginalized: tuple[int, ...],
    initial_carried: set[int],
    anchor_reason: dict[str, Any] | None,
    manifest_links: list[LinkFunction],
    loading_active: np.ndarray,
    residual_channels: np.ndarray,
    diffusion_active: np.ndarray,
    t0_active: np.ndarray,
) -> tuple[RBPFPartitionSpec, tuple[dict[str, Any], ...]]:
    carried = set(initial_carried)
    forced: list[dict[str, Any]] = []
    if anchor_reason is not None:
        anchor_idx = int(anchor_reason["latent_index"])
        carried.add(anchor_idx)
        forced.append(
            {
                "latent_index": anchor_idx,
                "latent_name": _latent_name(spec, anchor_idx),
                **anchor_reason,
            }
        )
    _force_residual_observation_latents(
        spec=spec,
        manifest_links=manifest_links,
        loading_active=loading_active,
        residual_channels=residual_channels,
        carried=carried,
        forced=forced,
    )
    changed = True
    while changed:
        changed = False
        if rbpf_mode == "independent":
            changed = (
                _force_independent_mixed_observation_latents(
                    spec=spec,
                    loading_active=loading_active,
                    carried=carried,
                    forced=forced,
                )
                or changed
            )
        changed = (
            _force_dynamics_closure(
                spec=spec,
                rbpf_mode=rbpf_mode,
                carried=carried,
                forced=forced,
            )
            or changed
        )
        changed = (
            _force_cross_covariance_latents(
                spec=spec,
                active=diffusion_active,
                carried=carried,
                forced=forced,
                reason="process_covariance_block",
            )
            or changed
        )
        changed = (
            _force_cross_covariance_latents(
                spec=spec,
                active=t0_active,
                carried=carried,
                forced=forced,
                reason="initial_covariance_block",
            )
            or changed
        )
    carried_tuple = tuple(idx for idx in range(spec.n_latent) if idx in carried)
    marginalized_tuple = tuple(
        idx for idx in candidate_marginalized if idx not in set(carried_tuple)
    )
    partition = RBPFPartitionSpec(
        carried_latent_indices=carried_tuple,
        marginalized_latent_indices=marginalized_tuple,
    )
    validate_rbpf_partition(partition, n_latent=spec.n_latent)
    return partition, tuple(forced)


def derive_rbpf_partition(
    *,
    spec,
    rbpf_mode: RBPFMode,
    marginalized_latent_indices: tuple[int, ...] | list[int] | None,
    manifest_links: list[LinkFunction],
    polya_gamma_channel_mask: jnp.ndarray,
) -> tuple[RBPFPartitionSpec, RBPFPartitionDiagnostics]:
    """Derive the maximal exact RBPF partition allowed by observations and dynamics."""
    mode = normalize_rbpf_mode(rbpf_mode)
    if mode == "none":
        partition = full_path_rbpf_partition(spec.n_latent)
        empty: tuple[int, ...] = ()
        diagnostics = RBPFPartitionDiagnostics(
            mode=mode,
            requested_marginalized_latent_indices=None
            if marginalized_latent_indices is None
            else tuple(int(idx) for idx in marginalized_latent_indices),
            candidate_marginalized_latent_indices=empty,
            carried_latent_indices=partition.carried_latent_indices,
            marginalized_latent_indices=partition.marginalized_latent_indices,
            forced_carried_latent_indices=empty,
            forced_carried=empty,
            gaussian_observation_channels=empty,
            polya_gamma_observation_channels=empty,
            residual_observation_channels=empty,
        )
        return partition, diagnostics

    candidate_marginalized = _validate_candidate_indices(
        spec.n_latent,
        marginalized_latent_indices,
    )
    candidate_set = set(candidate_marginalized)
    initial_carried = set(range(spec.n_latent)) - candidate_set
    loading_active = _is_active_entry(
        spec.lambda_block.free_support,
        spec.lambda_block.template,
    )
    gaussian_channels, pg_channels, residual_channels = _observation_channel_masks(
        spec,
        manifest_links,
        polya_gamma_channel_mask,
    )
    diffusion_active = _is_active_entry(
        spec.diffusion_block.diffusion_chol_support,
        spec.diffusion_block.diffusion_chol_template,
    )
    t0_corr_support = np.asarray(spec.t0_chol_block.correlation_support, dtype=bool)
    t0_corr_active = t0_corr_support | t0_corr_support.T
    t0_base_cov = np.asarray(spec.t0_chol_block.base_cov)
    t0_active = t0_corr_active | ~np.isclose(t0_base_cov, 0.0)

    residual_loaded = set(np.where(np.any(loading_active[residual_channels], axis=0))[0].tolist())
    candidates: list[tuple[RBPFPartitionSpec, tuple[dict[str, Any], ...]]] = []
    if initial_carried or residual_loaded:
        candidates.append(
            _derive_rbpf_partition_from_carried(
                spec=spec,
                rbpf_mode=mode,
                candidate_marginalized=candidate_marginalized,
                initial_carried=initial_carried,
                anchor_reason=None,
                manifest_links=manifest_links,
                loading_active=loading_active,
                residual_channels=residual_channels,
                diffusion_active=diffusion_active,
                t0_active=t0_active,
            )
        )
    else:
        for anchor_idx in candidate_marginalized:
            candidates.append(
                _derive_rbpf_partition_from_carried(
                    spec=spec,
                    rbpf_mode=mode,
                    candidate_marginalized=candidate_marginalized,
                    initial_carried=set(),
                    anchor_reason={
                        "latent_index": int(anchor_idx),
                        "reason": "runtime_requires_carried_latent",
                    },
                    manifest_links=manifest_links,
                    loading_active=loading_active,
                    residual_channels=residual_channels,
                    diffusion_active=diffusion_active,
                    t0_active=t0_active,
                )
            )
    if not candidates:
        raise ValueError("RBPF requires at least one latent dimension.")

    def _score(candidate: tuple[RBPFPartitionSpec, tuple[dict[str, Any], ...]]) -> tuple[int, int]:
        partition, _forced = candidate
        return (len(partition.marginalized_latent_indices), -partition.carried_latent_indices[0])

    partition, forced = max(candidates, key=_score)
    forced_carried_latent_indices = tuple(sorted({int(item["latent_index"]) for item in forced}))
    diagnostics = RBPFPartitionDiagnostics(
        mode=mode,
        requested_marginalized_latent_indices=None
        if marginalized_latent_indices is None
        else tuple(int(idx) for idx in marginalized_latent_indices),
        candidate_marginalized_latent_indices=candidate_marginalized,
        carried_latent_indices=partition.carried_latent_indices,
        marginalized_latent_indices=partition.marginalized_latent_indices,
        forced_carried_latent_indices=forced_carried_latent_indices,
        forced_carried=forced,
        gaussian_observation_channels=tuple(np.where(gaussian_channels)[0].astype(int).tolist()),
        polya_gamma_observation_channels=tuple(np.where(pg_channels)[0].astype(int).tolist()),
        residual_observation_channels=tuple(np.where(residual_channels)[0].astype(int).tolist()),
    )
    return partition, diagnostics


def validate_rbpf_partition(partition: RBPFPartitionSpec, *, n_latent: int) -> None:
    """Validate a partition before attaching it to a conditioned runtime."""
    all_indices = partition.carried_latent_indices + partition.marginalized_latent_indices
    if sorted(all_indices) != list(range(n_latent)):
        raise ValueError(
            "RBPF partition must cover each latent index exactly once; "
            f"got carried={partition.carried_latent_indices}, "
            f"marginalized={partition.marginalized_latent_indices}, n_latent={n_latent}."
        )
    if not partition.carried_latent_indices:
        raise ValueError("RBPF requires at least one carried latent dimension.")


def _enum_value_lower(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw).lower()


def _channel_param(
    extra_params: dict[str, jnp.ndarray] | None,
    key: str,
    *,
    default: float,
    n_channels: int,
    dtype,
) -> jnp.ndarray:
    params = extra_params or {}
    value = params.get(key, default)
    array = jnp.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return jnp.broadcast_to(array, (n_channels,))
    return jnp.broadcast_to(array, (n_channels,))


def _is_active_entry(free_support: np.ndarray, template: np.ndarray) -> np.ndarray:
    return np.asarray(free_support, dtype=bool) | ~np.isclose(np.asarray(template), 0.0)


def _cross_block_active(active: np.ndarray, left: tuple[int, ...], right: tuple[int, ...]) -> bool:
    return bool(np.any(active[np.ix_(left, right)])) or bool(np.any(active[np.ix_(right, left)]))


def _latent_group(idx: int, partition: RBPFPartitionSpec) -> str:
    if idx in partition.carried_latent_indices:
        return "carried"
    if idx in partition.marginalized_latent_indices:
        return "marginalized"
    raise ValueError(f"Latent index {idx} is not covered by RBPF partition {partition}.")


def _validate_rbpf_dynamics_components(spec, partition: RBPFPartitionSpec) -> bool:
    conditional = False
    for component in spec.dynamics_spec.components:
        if hasattr(component, "source") and hasattr(component, "target"):
            source_group = _latent_group(int(component.source), partition)
            target_group = _latent_group(int(component.target), partition)
            is_linear = type(component).__name__ == "LinearEdgeSpec"
            if target_group == "marginalized" and not is_linear:
                raise NotImplementedError(
                    "RBPF marginalized dynamics must remain linear-Gaussian; "
                    f"component {component!r} has a marginalized target."
                )
            if source_group == "marginalized" and target_group == "carried":
                raise NotImplementedError(
                    "RBPF marginalization requires carried dynamics not to depend on "
                    f"marginalized latents; component {component!r} violates this."
                )
            if source_group == "carried" and target_group == "marginalized":
                conditional = True
        if (
            hasattr(component, "source_a")
            and hasattr(component, "source_b")
            and hasattr(component, "target")
        ):
            groups = {
                _latent_group(int(component.source_a), partition),
                _latent_group(int(component.source_b), partition),
                _latent_group(int(component.target), partition),
            }
            if "marginalized" in groups:
                raise NotImplementedError(
                    "RBPF marginalized dynamics must remain linear-Gaussian; "
                    f"multiplicative component {component!r} touches marginalized latents."
                )
    return conditional


def build_gaussian_rbpf_observation_plan(
    spec,
    partition: RBPFPartitionSpec,
    manifest_links: list[LinkFunction],
    polya_gamma_channel_mask: jnp.ndarray,
) -> RBPFObservationPlan:
    """Validate and identify rows for the Gaussian RBPF adapter."""
    validate_rbpf_partition(partition, n_latent=spec.n_latent)
    empty_mask = jnp.zeros((spec.n_manifest,), dtype=bool)
    if partition.is_full_path:
        return RBPFObservationPlan(
            channel_mask=empty_mask,
            gaussian_channel_mask=empty_mask,
            polya_gamma_channel_mask=empty_mask,
            negative_binomial_polya_gamma_channel_mask=empty_mask,
            structure="none",
        )

    carried = partition.carried_latent_indices
    marginalized = partition.marginalized_latent_indices
    conditional = _validate_rbpf_dynamics_components(spec, partition)

    diffusion_active = _is_active_entry(
        spec.diffusion_block.diffusion_chol_support,
        spec.diffusion_block.diffusion_chol_template,
    )
    diffusion_cross_active = _cross_block_active(diffusion_active, carried, marginalized)
    if diffusion_cross_active:
        raise NotImplementedError(
            "RBPF marginalization currently requires block-diagonal instantaneous process diffusion."
        )

    t0_corr_support = np.asarray(spec.t0_chol_block.correlation_support, dtype=bool)
    t0_corr_active = t0_corr_support | t0_corr_support.T
    t0_base_cov = np.asarray(spec.t0_chol_block.base_cov)
    t0_base_active = ~np.isclose(t0_base_cov, 0.0)
    if _cross_block_active(t0_corr_active | t0_base_active, carried, marginalized):
        raise NotImplementedError(
            "RBPF marginalization currently requires block-diagonal initial-state covariance."
        )

    loading_active = _is_active_entry(
        spec.lambda_block.free_support,
        spec.lambda_block.template,
    )
    uses_carried = np.any(loading_active[:, carried], axis=1)
    uses_marginalized = np.any(loading_active[:, marginalized], axis=1)
    pg_channels = np.asarray(polya_gamma_channel_mask, dtype=bool)
    if pg_channels.shape != (spec.n_manifest,):
        raise ValueError(
            "polya_gamma_channel_mask must have shape "
            f"({spec.n_manifest},), got {pg_channels.shape}."
        )
    gaussian_channels = np.zeros((spec.n_manifest,), dtype=bool)
    rbpf_pg_channels = np.zeros((spec.n_manifest,), dtype=bool)
    rbpf_pg_negative_binomial_channels = np.zeros((spec.n_manifest,), dtype=bool)
    for channel_idx in np.where(uses_marginalized)[0].tolist():
        dist_value = _enum_value_lower(spec.manifest_dists[channel_idx])
        link_value = _enum_value_lower(manifest_links[channel_idx])
        is_gaussian_identity = dist_value == _enum_value_lower(
            DistributionFamily.GAUSSIAN
        ) and link_value == _enum_value_lower(LinkFunction.IDENTITY)
        if is_gaussian_identity:
            gaussian_channels[channel_idx] = True
            continue
        if bool(pg_channels[channel_idx]):
            rbpf_pg_channels[channel_idx] = True
            rbpf_pg_negative_binomial_channels[channel_idx] = dist_value == _enum_value_lower(
                DistributionFamily.NEGATIVE_BINOMIAL
            )
            continue
        raise NotImplementedError(
            "RBPF marginalization currently consumes Gaussian identity rows or "
            "PG-conditioned affine-logit rows; "
            f"channel {channel_idx} has distribution={dist_value}, link={link_value}."
        )
    marginal_channels = gaussian_channels | rbpf_pg_channels
    if bool(np.any(rbpf_pg_channels)):
        manifest_chol = np.asarray(spec.manifest_chol_block.template)
        manifest_cov = manifest_chol @ manifest_chol.T
        pg_indices = np.where(rbpf_pg_channels)[0]
        gaussian_indices = np.where(gaussian_channels)[0]
        if (
            pg_indices.size
            and gaussian_indices.size
            and bool(np.any(~np.isclose(manifest_cov[np.ix_(pg_indices, gaussian_indices)], 0.0)))
        ):
            raise NotImplementedError(
                "RBPF PG-conditioned rows require zero manifest-noise covariance "
                "with Gaussian RBPF rows."
            )
    conditional = conditional or bool(np.any(uses_carried & uses_marginalized))

    return RBPFObservationPlan(
        channel_mask=jnp.asarray(marginal_channels, dtype=bool),
        gaussian_channel_mask=jnp.asarray(gaussian_channels, dtype=bool),
        polya_gamma_channel_mask=jnp.asarray(rbpf_pg_channels, dtype=bool),
        negative_binomial_polya_gamma_channel_mask=jnp.asarray(
            rbpf_pg_negative_binomial_channels,
            dtype=bool,
        ),
        structure="conditional" if conditional else "independent",
    )


def validate_rbpf_mode(
    rbpf_mode: RBPFMode,
    partition: RBPFPartitionSpec,
    observation_plan: RBPFObservationPlan,
) -> None:
    """Require the explicit requested RBPF mode to match the validated structure."""
    if rbpf_mode == "none":
        if not partition.is_full_path:
            raise ValueError(
                "rbpf_mode='none' cannot be combined with rbpf_marginalized_latent_indices."
            )
        if observation_plan.structure != "none":
            raise ValueError("rbpf_mode='none' requires an unpartitioned full-path runtime.")
        return

    if partition.is_full_path:
        if observation_plan.structure != "none":
            raise ValueError("Full-path RBPF partition requires an empty observation plan.")
        return
    if rbpf_mode == "conditional" and observation_plan.structure in {"independent", "conditional"}:
        return
    if observation_plan.structure != rbpf_mode:
        raise ValueError(
            f"rbpf_mode={rbpf_mode!r} does not match the validated RBPF structure "
            f"{observation_plan.structure!r}. Choose the matching mode explicitly."
        )


def mask_rbpf_observations(plan: RBPFObservationPlan, observations: jnp.ndarray) -> jnp.ndarray:
    """Remove RBPF-consumed observations from the carried-state residual likelihood."""
    return jnp.where(plan.channel_mask[None, :], jnp.nan, observations)


def _take_square(mats: jnp.ndarray, indices: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(jnp.take(mats, indices, axis=-2), indices, axis=-1)


def _take_rect(mats: jnp.ndarray, rows: jnp.ndarray, cols: jnp.ndarray) -> jnp.ndarray:
    return jnp.take(jnp.take(mats, rows, axis=-2), cols, axis=-1)


def build_rbpf_marginal_context(
    *,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    H: jnp.ndarray,
    d_meas: jnp.ndarray,
    R: jnp.ndarray,
    partition: RBPFPartitionSpec,
    observation_plan: RBPFObservationPlan,
    H_rows: jnp.ndarray | None = None,
    d_meas_rows: jnp.ndarray | None = None,
    extra_params: dict[str, jnp.ndarray] | None = None,
) -> RBPFMarginalContext:
    """Project the full linear-Gaussian context onto carried/marginalized blocks."""
    carried = jnp.asarray(partition.carried_latent_indices, dtype=jnp.int32)
    marginalized = jnp.asarray(partition.marginalized_latent_indices, dtype=jnp.int32)
    observation_indices = jnp.nonzero(
        observation_plan.channel_mask,
        size=int(np.sum(np.asarray(observation_plan.channel_mask))),
    )[0].astype(jnp.int32)
    gaussian_row_mask = jnp.take(observation_plan.gaussian_channel_mask, observation_indices)
    pg_row_mask = jnp.take(observation_plan.polya_gamma_channel_mask, observation_indices)
    nb_pg_row_mask = jnp.take(
        observation_plan.negative_binomial_polya_gamma_channel_mask,
        observation_indices,
    )
    obs_r = jnp.take(
        _channel_param(
            extra_params,
            "obs_r",
            default=5.0,
            n_channels=int(H.shape[0]),
            dtype=H.dtype,
        ),
        observation_indices,
        axis=0,
    )
    init_pred_mean = Ad[0] @ init_mean + cd[0]
    init_pred_cov = symmetrize_with_jitter(Ad[0] @ init_cov @ Ad[0].T + Qd[0], jitter=0.0)
    H_obs = jnp.take(H, observation_indices, axis=0)
    H_m = jnp.take(H_obs, marginalized, axis=1)
    H_c = jnp.take(H_obs, carried, axis=1)
    if H_rows is None:
        H_m_rows = None
        H_c_rows = None
    else:
        H_rows_obs = jnp.take(H_rows, observation_indices, axis=1)
        H_m_rows = jnp.take(H_rows_obs, marginalized, axis=2)
        H_c_rows = jnp.take(H_rows_obs, carried, axis=2)
    if d_meas_rows is None:
        d_rows = None
    else:
        d_rows = jnp.take(d_meas_rows, observation_indices, axis=1)

    return RBPFMarginalContext(
        Ad_cc=_take_square(Ad, carried),
        Ad_mm=_take_square(Ad, marginalized),
        Ad_mc=_take_rect(Ad, marginalized, carried),
        Q_cc=_take_square(Qd, carried),
        Q_mm=_take_square(Qd, marginalized),
        Q_mc=_take_rect(Qd, marginalized, carried),
        cd_c=jnp.take(cd, carried, axis=-1),
        cd_m=jnp.take(cd, marginalized, axis=-1),
        init_mean_c=jnp.take(init_pred_mean, carried, axis=-1),
        init_mean_m=jnp.take(init_pred_mean, marginalized, axis=-1),
        init_cov_cc=_take_square(init_pred_cov, carried),
        init_cov_mm=_take_square(init_pred_cov, marginalized),
        init_cov_mc=_take_rect(init_pred_cov, marginalized, carried),
        H_m=H_m,
        H_c=H_c,
        H_m_rows=H_m_rows,
        H_c_rows=H_c_rows,
        d_meas=jnp.take(d_meas, observation_indices, axis=0),
        d_meas_rows=d_rows,
        R=_take_square(R, observation_indices),
        observation_indices=observation_indices,
        gaussian_row_mask=gaussian_row_mask,
        polya_gamma_row_mask=pg_row_mask,
        negative_binomial_polya_gamma_row_mask=nb_pg_row_mask,
        obs_r=obs_r,
    )


def reduce_context_to_carried(
    *,
    Ad: jnp.ndarray,
    Qd: jnp.ndarray,
    cd: jnp.ndarray,
    init_mean: jnp.ndarray,
    init_cov: jnp.ndarray,
    H: jnp.ndarray,
    partition: RBPFPartitionSpec,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project state-transition and loading matrices onto carried dimensions."""
    carried = jnp.asarray(partition.carried_latent_indices, dtype=jnp.int32)
    return (
        _take_square(Ad, carried),
        _take_square(Qd, carried),
        jnp.take(cd, carried, axis=-1),
        jnp.take(init_mean, carried, axis=-1),
        _take_square(init_cov, carried),
        jnp.take(H, carried, axis=1),
    )


def _solve_spd(mat: jnp.ndarray, rhs: jnp.ndarray) -> jnp.ndarray:
    return jla.solve(mat, rhs, assume_a="pos")


def _condition_marginal_initial(
    marginal_context: RBPFMarginalContext,
    carried_t: jnp.ndarray,
    *,
    jitter: float,
) -> RBPFMarginalFilterState:
    dtype = carried_t.dtype
    init_mean_c = marginal_context.init_mean_c.astype(dtype)
    init_mean_m = marginal_context.init_mean_m.astype(dtype)
    init_cov_cc = marginal_context.init_cov_cc.astype(dtype)
    init_cov_mc = marginal_context.init_cov_mc.astype(dtype)
    init_cov_mm = marginal_context.init_cov_mm.astype(dtype)
    cov_cc = symmetrize_with_jitter(init_cov_cc, jitter=jitter)
    diff_c = carried_t - init_mean_c
    mean = init_mean_m + init_cov_mc @ _solve_spd(cov_cc, diff_c)
    cov = init_cov_mm - init_cov_mc @ _solve_spd(
        cov_cc,
        init_cov_mc.T,
    )
    return RBPFMarginalFilterState(
        mean=mean,
        cov=symmetrize_with_jitter(cov, jitter=jitter),
    )


def _rbpf_observation_at(
    marginal_context: RBPFMarginalContext,
    observations: jnp.ndarray,
    observation_auxiliary,
    time_idx: jnp.ndarray,
    *,
    dtype,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    y_raw = jnp.take(observations[time_idx], marginal_context.observation_indices, axis=0)
    observed_mask = ~jnp.isnan(y_raw)
    y_raw = jnp.nan_to_num(y_raw, nan=0.0).astype(dtype)
    row_pg_mask = marginal_context.polya_gamma_row_mask.astype(dtype)
    row_gaussian_mask = marginal_context.gaussian_row_mask.astype(dtype)
    row_nb_pg_mask_bool = marginal_context.negative_binomial_polya_gamma_row_mask.astype(bool)
    if observation_auxiliary is None:
        omega_rows = jnp.ones_like(y_raw)
        kappa_rows = y_raw
        offset_rows = jnp.zeros_like(y_raw)
        active_pg_mask = observed_mask
        gamma_base_terms = jnp.zeros((*y_raw.shape, 0), dtype=dtype)
    else:
        omega_t = jnp.take(
            observation_auxiliary.omega[time_idx],
            marginal_context.observation_indices,
            axis=0,
        )
        observed_values_t = jnp.take(
            observation_auxiliary.observed_values[time_idx],
            marginal_context.observation_indices,
            axis=0,
        )
        active_t = jnp.take(
            observation_auxiliary.active_mask[time_idx],
            marginal_context.observation_indices,
            axis=0,
        )
        gamma_base_terms = jnp.take(
            observation_auxiliary.gamma_base_terms[time_idx],
            marginal_context.observation_indices,
            axis=0,
        )
        omega_rows = omega_t.astype(dtype)
        active_pg_mask = active_t > 0
        r_rows = marginal_context.obs_r.astype(dtype)
        nb_kappa = 0.5 * (observed_values_t.astype(dtype) - r_rows)
        nb_offset = -jnp.log(jnp.maximum(r_rows, jnp.asarray(1e-8, dtype=dtype)))
        bernoulli_kappa = observed_values_t.astype(dtype) - 0.5
        kappa_rows = jnp.where(row_nb_pg_mask_bool, nb_kappa, bernoulli_kappa)
        offset_rows = jnp.where(row_nb_pg_mask_bool, nb_offset, 0.0)

    omega_safe = jnp.maximum(omega_rows, jnp.asarray(1e-8, dtype=dtype))
    pg_y = kappa_rows / omega_safe - offset_rows
    y_t = jnp.where(marginal_context.polya_gamma_row_mask, pg_y, y_raw)
    row_observed_mask = jnp.where(
        marginal_context.polya_gamma_row_mask,
        active_pg_mask,
        observed_mask,
    )
    row_observed = row_observed_mask.astype(dtype)
    gaussian_outer = row_gaussian_mask[:, None] * row_gaussian_mask[None, :]
    R_gaussian = marginal_context.R.astype(dtype) * gaussian_outer
    R_pg = jnp.diag(row_pg_mask / omega_safe)
    R_full = R_gaussian + R_pg
    observed_outer = row_observed[:, None] * row_observed[None, :]
    missing_diag = jnp.diag(1.0 - row_observed)
    R_masked = R_full * observed_outer + missing_diag
    pg_active_mask = marginal_context.polya_gamma_row_mask & row_observed_mask
    pg_correction = polya_gamma_gaussian_logpdf_correction(
        kappa_rows,
        omega_rows,
        pg_active_mask,
    )
    nb_base_terms = negative_binomial_finite_sum_base_log_terms(
        y_raw,
        marginal_context.obs_r.astype(dtype),
        gamma_base_terms.astype(dtype),
        row_nb_pg_mask_bool & row_observed_mask,
    )
    correction = jnp.sum(pg_correction + nb_base_terms)
    return jnp.where(row_observed_mask, y_t, 0.0), R_masked, row_observed_mask, correction


def _rbpf_design_at(
    marginal_context: RBPFMarginalContext,
    time_idx: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if marginal_context.H_m_rows is None:
        return marginal_context.H_m, marginal_context.H_c, marginal_context.d_meas
    assert marginal_context.H_c_rows is not None
    assert marginal_context.d_meas_rows is not None
    return (
        marginal_context.H_m_rows[time_idx],
        marginal_context.H_c_rows[time_idx],
        marginal_context.d_meas_rows[time_idx],
    )


def _rbpf_gaussian_update(
    marginal_context: RBPFMarginalContext,
    filter_state: RBPFMarginalFilterState,
    observations_t: jnp.ndarray,
    observation_cov_t: jnp.ndarray,
    row_observed_mask: jnp.ndarray,
    carried_t: jnp.ndarray,
    time_idx: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[RBPFMarginalFilterState, jnp.ndarray]:
    dtype = carried_t.dtype
    H_m, H_c, d_meas = _rbpf_design_at(marginal_context, time_idx)
    if H_m.shape[0] == 0:
        return filter_state, jnp.asarray(0.0, dtype=dtype)
    H_m = H_m.astype(dtype)
    H_c = H_c.astype(dtype)
    d_meas = d_meas.astype(dtype)
    mean = filter_state.mean.astype(dtype)
    cov = filter_state.cov.astype(dtype)
    observations_t = observations_t.astype(dtype)
    observation_cov_t = observation_cov_t.astype(dtype)
    observed = row_observed_mask.astype(dtype)
    H_m = H_m * observed[:, None]
    H_c = H_c * observed[:, None]
    d_meas = d_meas * observed
    resid = observations_t - (H_m @ mean + H_c @ carried_t + d_meas)
    innovation_cov = symmetrize_with_jitter(
        H_m @ cov @ H_m.T + observation_cov_t,
        jitter=jitter,
    )
    chol = jnp.linalg.cholesky(innovation_cov)
    whitened = jla.solve_triangular(chol, resid, lower=True)
    logdet = 2.0 * jnp.sum(jnp.log(jnp.clip(jnp.diag(chol), 1e-12)))
    n_observed = jnp.sum(observed)
    loglik = -0.5 * (
        n_observed * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=observations_t.dtype))
        + logdet
        + whitened @ whitened
    )
    PHt = cov @ H_m.T
    gain_t = jla.solve_triangular(chol, PHt.T, lower=True)
    gain = jla.solve_triangular(chol.T, gain_t, lower=False).T
    updated_mean = mean + gain @ resid
    updated_cov = symmetrize_with_jitter(
        cov - gain @ H_m @ cov,
        jitter=jitter,
    )
    return RBPFMarginalFilterState(mean=updated_mean, cov=updated_cov), loglik


def rbpf_initial_filter_update(
    marginal_context: RBPFMarginalContext | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_t: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> tuple[RBPFMarginalFilterState | None, jnp.ndarray]:
    """Initialize and update marginalized filter at t=0 for one carried state."""
    if marginal_context is None or marginal_context.H_m.shape[0] == 0:
        return None, jnp.asarray(0.0, dtype=carried_t.dtype)
    y0, R0, observed0, correction0 = _rbpf_observation_at(
        marginal_context,
        observations,
        observation_auxiliary,
        jnp.asarray(0, dtype=jnp.int32),
        dtype=carried_t.dtype,
    )
    initial_state = _condition_marginal_initial(marginal_context, carried_t, jitter=jitter)
    updated_state, loglik = _rbpf_gaussian_update(
        marginal_context,
        initial_state,
        y0,
        R0,
        observed0,
        carried_t,
        jnp.asarray(0, dtype=jnp.int32),
        jitter=jitter,
    )
    return updated_state, jnp.asarray(loglik + correction0, dtype=carried_t.dtype)


def rbpf_step_filter_update(
    marginal_context: RBPFMarginalContext | None,
    previous_filter_state: RBPFMarginalFilterState | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_prev: jnp.ndarray,
    carried_t: jnp.ndarray,
    time_idx: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> tuple[RBPFMarginalFilterState | None, jnp.ndarray]:
    """Predict/update marginalized filter for one carried transition."""
    if marginal_context is None or marginal_context.H_m.shape[0] == 0:
        return None, jnp.asarray(0.0, dtype=carried_t.dtype)
    assert previous_filter_state is not None
    pred_mean, pred_cov, _offset = _rbpf_transition_condition(
        marginal_context,
        previous_filter_state,
        carried_prev,
        carried_t,
        time_idx,
        jitter=jitter,
    )
    y_t, R_t, observed_t, correction_t = _rbpf_observation_at(
        marginal_context,
        observations,
        observation_auxiliary,
        time_idx,
        dtype=carried_t.dtype,
    )
    updated_state, loglik = _rbpf_gaussian_update(
        marginal_context,
        RBPFMarginalFilterState(mean=pred_mean, cov=pred_cov),
        y_t,
        R_t,
        observed_t,
        carried_t,
        time_idx,
        jitter=jitter,
    )
    return updated_state, jnp.asarray(loglik + correction_t, dtype=carried_t.dtype)


def rbpf_marginal_log_likelihood(
    marginal_context: RBPFMarginalContext | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_trajectory: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """Kalman marginal likelihood of the marginalized subsystem conditional on carried path."""
    if marginal_context is None:
        return jnp.asarray(0.0, dtype=carried_trajectory.dtype)
    if marginal_context.H_m.shape[0] == 0:
        return jnp.asarray(0.0, dtype=carried_trajectory.dtype)

    filter0, init_loglik = rbpf_initial_filter_update(
        marginal_context,
        observations,
        observation_auxiliary,
        carried_trajectory[0],
        jitter=jitter,
    )

    def _step(carry, inputs):
        previous_filter, carried_prev = carry
        carried_t, time_idx = inputs
        next_filter, loglik = rbpf_step_filter_update(
            marginal_context,
            previous_filter,
            observations,
            observation_auxiliary,
            carried_prev,
            carried_t,
            time_idx,
            jitter=jitter,
        )
        return (next_filter, carried_t), loglik

    if carried_trajectory.shape[0] == 1:
        return jnp.asarray(init_loglik, dtype=carried_trajectory.dtype)
    _, loglik_rest = jax.lax.scan(
        _step,
        (filter0, carried_trajectory[0]),
        (
            carried_trajectory[1:],
            jnp.arange(1, carried_trajectory.shape[0], dtype=jnp.int32),
        ),
    )
    return jnp.asarray(init_loglik + jnp.sum(loglik_rest), dtype=carried_trajectory.dtype)


def rbpf_marginal_log_likelihoods(
    marginal_context: RBPFMarginalContext | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_trajectory: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """Per-time Kalman marginal likelihoods conditional on a carried path."""
    if marginal_context is None or marginal_context.H_m.shape[0] == 0:
        return jnp.zeros((carried_trajectory.shape[0],), dtype=carried_trajectory.dtype)
    filter0, init_loglik = rbpf_initial_filter_update(
        marginal_context,
        observations,
        observation_auxiliary,
        carried_trajectory[0],
        jitter=jitter,
    )

    def _step(carry, inputs):
        previous_filter, carried_prev = carry
        carried_t, time_idx = inputs
        next_filter, loglik = rbpf_step_filter_update(
            marginal_context,
            previous_filter,
            observations,
            observation_auxiliary,
            carried_prev,
            carried_t,
            time_idx,
            jitter=jitter,
        )
        return (next_filter, carried_t), loglik

    if carried_trajectory.shape[0] == 1:
        return jnp.reshape(init_loglik, (1,))
    _, loglik_rest = jax.lax.scan(
        _step,
        (filter0, carried_trajectory[0]),
        (
            carried_trajectory[1:],
            jnp.arange(1, carried_trajectory.shape[0], dtype=jnp.int32),
        ),
    )
    return jnp.concatenate([jnp.reshape(init_loglik, (1,)), loglik_rest])


def _sample_gaussian(key: jax.Array, mean: jnp.ndarray, cov: jnp.ndarray, *, jitter: float):
    chol = jnp.linalg.cholesky(symmetrize_with_jitter(cov, jitter=jitter))
    eps = jax.random.normal(key, mean.shape, dtype=mean.dtype)
    return mean + chol @ eps


def _rbpf_transition_condition(
    marginal_context: RBPFMarginalContext,
    previous_filter_state: RBPFMarginalFilterState,
    carried_prev: jnp.ndarray,
    carried_t: jnp.ndarray,
    time_idx: jnp.ndarray,
    *,
    jitter: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = carried_t.dtype
    Ad_cc_t = marginal_context.Ad_cc[time_idx].astype(dtype)
    Ad_mm_t = marginal_context.Ad_mm[time_idx].astype(dtype)
    Ad_mc_t = marginal_context.Ad_mc[time_idx].astype(dtype)
    Q_cc_t = symmetrize_with_jitter(marginal_context.Q_cc[time_idx].astype(dtype), jitter=jitter)
    Q_mm_t = marginal_context.Q_mm[time_idx].astype(dtype)
    Q_mc_t = marginal_context.Q_mc[time_idx].astype(dtype)
    cd_c_t = marginal_context.cd_c[time_idx].astype(dtype)
    cd_m_t = marginal_context.cd_m[time_idx].astype(dtype)
    previous_mean = previous_filter_state.mean.astype(dtype)
    previous_cov = previous_filter_state.cov.astype(dtype)

    carried_innovation = carried_t - (Ad_cc_t @ carried_prev + cd_c_t)
    noise_mean = Q_mc_t @ _solve_spd(Q_cc_t, carried_innovation)
    noise_cov = Q_mm_t - Q_mc_t @ _solve_spd(Q_cc_t, Q_mc_t.T)
    transition_offset = Ad_mc_t @ carried_prev + cd_m_t + noise_mean
    pred_mean = Ad_mm_t @ previous_mean + transition_offset
    pred_cov = symmetrize_with_jitter(
        Ad_mm_t @ previous_cov @ Ad_mm_t.T + noise_cov,
        jitter=jitter,
    )
    return pred_mean, pred_cov, transition_offset


def sample_rbpf_marginal_trajectory(
    key: jax.Array,
    marginal_context: RBPFMarginalContext | None,
    observations: jnp.ndarray,
    observation_auxiliary,
    carried_trajectory: jnp.ndarray,
    *,
    jitter: float = 1e-6,
) -> jnp.ndarray:
    """Sample marginalized latent path conditional on carried path and Gaussian/PG rows."""
    if marginal_context is None:
        return jnp.zeros((carried_trajectory.shape[0], 0), dtype=carried_trajectory.dtype)
    n_marginalized = int(marginal_context.Ad_mm.shape[-1])
    if n_marginalized == 0:
        return jnp.zeros((carried_trajectory.shape[0], 0), dtype=carried_trajectory.dtype)

    filter0, _init_loglik = rbpf_initial_filter_update(
        marginal_context,
        observations,
        observation_auxiliary,
        carried_trajectory[0],
        jitter=jitter,
    )
    assert filter0 is not None

    def _forward_step(carry, inputs):
        previous_filter, carried_prev = carry
        carried_t, time_idx = inputs
        pred_mean, pred_cov, _offset = _rbpf_transition_condition(
            marginal_context,
            previous_filter,
            carried_prev,
            carried_t,
            time_idx,
            jitter=jitter,
        )
        y_t, R_t, observed_t, _correction_t = _rbpf_observation_at(
            marginal_context,
            observations,
            observation_auxiliary,
            time_idx,
            dtype=carried_t.dtype,
        )
        next_filter, _loglik = _rbpf_gaussian_update(
            marginal_context,
            RBPFMarginalFilterState(mean=pred_mean, cov=pred_cov),
            y_t,
            R_t,
            observed_t,
            carried_t,
            time_idx,
            jitter=jitter,
        )
        return (next_filter, carried_t), next_filter

    if carried_trajectory.shape[0] == 1:
        filter_means = filter0.mean[None, :]
        filter_covs = filter0.cov[None, :, :]
    else:
        _, filter_rest = jax.lax.scan(
            _forward_step,
            (filter0, carried_trajectory[0]),
            (
                carried_trajectory[1:],
                jnp.arange(1, carried_trajectory.shape[0], dtype=jnp.int32),
            ),
        )
        filter_means = jnp.concatenate([filter0.mean[None, :], filter_rest.mean], axis=0)
        filter_covs = jnp.concatenate([filter0.cov[None, :, :], filter_rest.cov], axis=0)

    keys = jax.random.split(key, carried_trajectory.shape[0])
    final_sample = _sample_gaussian(
        keys[-1],
        filter_means[-1],
        filter_covs[-1],
        jitter=jitter,
    )

    def _backward_step(next_sample, inputs):
        filter_mean_t, filter_cov_t, carried_t, carried_next, time_idx, sample_key = inputs
        dtype = next_sample.dtype
        filter_mean_t = filter_mean_t.astype(dtype)
        filter_cov_t = filter_cov_t.astype(dtype)
        carried_t = carried_t.astype(dtype)
        carried_next = carried_next.astype(dtype)
        filter_state_t = RBPFMarginalFilterState(mean=filter_mean_t, cov=filter_cov_t)
        pred_mean, pred_cov, transition_offset = _rbpf_transition_condition(
            marginal_context,
            filter_state_t,
            carried_t,
            carried_next,
            time_idx,
            jitter=jitter,
        )
        F_t = marginal_context.Ad_mm[time_idx].astype(dtype)
        eye = jnp.eye(pred_cov.shape[0], dtype=dtype)
        smoother_gain = filter_cov_t @ F_t.T @ _solve_spd(pred_cov, eye)
        smooth_mean = filter_mean_t + smoother_gain @ (next_sample - pred_mean)
        smooth_cov = filter_cov_t - smoother_gain @ pred_cov @ smoother_gain.T
        del transition_offset
        sample_t = _sample_gaussian(sample_key, smooth_mean, smooth_cov, jitter=jitter)
        return sample_t, sample_t

    if carried_trajectory.shape[0] == 1:
        return final_sample[None, :]
    _, reversed_samples = jax.lax.scan(
        _backward_step,
        final_sample,
        (
            filter_means[:-1][::-1],
            filter_covs[:-1][::-1],
            carried_trajectory[:-1][::-1],
            carried_trajectory[1:][::-1],
            jnp.arange(1, carried_trajectory.shape[0], dtype=jnp.int32)[::-1],
            keys[:-1][::-1],
        ),
    )
    return jnp.concatenate([reversed_samples[::-1], final_sample[None, :]], axis=0)
