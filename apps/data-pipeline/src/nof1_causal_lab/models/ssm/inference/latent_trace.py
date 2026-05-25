"""Shared latent-kernel trace diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

MH_LOG_ALPHA_FIELD_NAMES = (
    "log_alpha_per_t",
    "log_alpha_obs_per_t",
    "log_alpha_fwd_minus_rev_per_t",
    "log_alpha_q_per_t",
    "log_alpha_global",
    "log_alpha",
)

LATENT_MOVE_FIELD_NAMES = (
    "latent_move_rms",
    "latent_move_max_abs",
    "latent_move_rms_per_t",
)

PIT_PARTICLE_TRACE_FIELD_NAMES = (
    "pit_updated_mask_per_t",
    "pit_selected_origin_per_t",
    "pit_proposal_grad_norm_per_t",
    "pit_proposal_shift_norm_per_t",
    "pit_auxiliary_noise_rms_per_t",
    "pit_delta_per_t",
    "pit_resampling_entropy_per_t",
    "pit_resampling_ess_per_t",
    "pit_resampling_log_normalizer_per_t",
    "particle_mgrad_updated_mask_per_t",
    "particle_mgrad_selected_origin_per_t",
    "particle_mgrad_ref_grad_norm_per_t",
    "particle_mgrad_auxiliary_shift_norm_per_t",
    "particle_mgrad_auxiliary_noise_rms_per_t",
    "particle_mgrad_delta_per_t",
    "particle_mgrad_resampling_entropy_per_t",
    "particle_mgrad_resampling_ess_per_t",
    "particle_mgrad_resampling_log_normalizer_per_t",
)


@dataclass(frozen=True)
class LatentTraceConfig:
    """Trace flags shared by latent-kernel method wrappers."""

    emit_per_t_log_alpha: bool = False
    debug_particle_trace: bool = False


def validate_latent_trace_config(method_name: str, config: LatentTraceConfig) -> None:
    """Reject trace modes that do not exist for the selected latent kernel."""
    if method_name == "pit_particle_mgrad" and config.emit_per_t_log_alpha:
        raise ValueError(
            "emit_per_t_log_alpha is only supported for auxiliary-Kalman MH latent "
            "kernels; use debug_particle_trace for particle latent per-time "
            "particle diagnostics."
        )
    if method_name == "aux_kalman_mcmc" and config.debug_particle_trace:
        raise ValueError(
            "debug_particle_trace is only supported for particle latent kernels; use "
            "emit_per_t_log_alpha for auxiliary-Kalman MH log-alpha diagnostics."
        )


def build_latent_trace_diagnostics(
    chain_extra_fields: Mapping[str, Any],
    config: LatentTraceConfig,
) -> dict[str, Any]:
    """Summarize which latent trace fields were requested and emitted."""
    log_alpha_fields = [
        field_name for field_name in MH_LOG_ALPHA_FIELD_NAMES if field_name in chain_extra_fields
    ]
    latent_move_fields = [
        field_name for field_name in LATENT_MOVE_FIELD_NAMES if field_name in chain_extra_fields
    ]
    particle_trace_fields = [
        field_name
        for field_name in PIT_PARTICLE_TRACE_FIELD_NAMES
        if field_name in chain_extra_fields
    ]
    return {
        "emit_per_t_log_alpha": bool(config.emit_per_t_log_alpha),
        "debug_particle_trace": bool(config.debug_particle_trace),
        "log_alpha_fields": log_alpha_fields,
        "latent_move_fields": latent_move_fields,
        "particle_trace_fields": particle_trace_fields,
        "emitted_fields": log_alpha_fields + latent_move_fields + particle_trace_fields,
    }
