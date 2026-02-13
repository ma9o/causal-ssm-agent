"""Stage 5 GPU dispatch via Modal.

Runs fit_model + power_scaling + interventions together on a single
Modal GPU container so JAX arrays never cross the serialization boundary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]  # project root


def _make_image(gpu: str):
    """Build a Modal image for JAX + CUDA with the project installed.

    Mirrors benchmarks/modal_infra.py.
    """
    import modal

    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("git")
        .pip_install("uv")
        .uv_sync(uv_project_dir=str(ROOT), groups=["dev"], frozen=True)
        .uv_pip_install("jax[cuda12]", gpu=gpu)
        .env({"PYTHONPATH": "/root"})
        .add_local_dir(ROOT / "src" / "causal_ssm_agent", remote_path="/root/causal_ssm_agent")
    )


def _stage5_on_gpu(
    stage4_result: dict,
    data_bytes: bytes,
    sampler_config: dict | None,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None,
) -> dict[str, Any]:
    """Execute all stage 5 tasks inside a GPU container.

    This function runs *remotely* on Modal. All inputs/outputs are plain
    Python types (no JAX arrays cross the boundary).
    """
    import io

    import jax.numpy as jnp
    import polars as pl_inner
    from jax import vmap

    from causal_ssm_agent.models.ssm.counterfactual import treatment_effect
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
    from causal_ssm_agent.utils.parametric_id import power_scaling_sensitivity

    # ---------- reconstruct data ----------
    raw_data = pl_inner.read_ipc(io.BytesIO(data_bytes))

    # ---------- fit ----------
    model_spec = stage4_result.get("model_spec", {})
    priors = stage4_result.get("priors", {})

    builder = SSMModelBuilder(model_spec=model_spec, priors=priors, sampler_config=sampler_config)

    if raw_data.is_empty():
        return {
            "ps_result": {"checked": False, "error": "No data"},
            "intervention_results": [],
        }

    time_col = "time_bucket" if "time_bucket" in raw_data.columns else "timestamp"
    wide_data = (
        raw_data.with_columns(pl_inner.col("value").cast(pl_inner.Float64, strict=False))
        .pivot(on="indicator", index=time_col, values="value")
        .sort(time_col)
    )

    if wide_data.schema[time_col] in (pl_inner.Datetime, pl_inner.Date):
        t0 = wide_data[time_col].min()
        wide_data = wide_data.with_columns(
            ((pl_inner.col(time_col) - t0).dt.total_seconds() / 86400.0).alias(time_col)
        )

    X = wide_data.to_pandas()
    if time_col in X.columns:
        X = X.rename(columns={time_col: "time"})

    result = builder.fit(X)
    print(f"Fit complete: method={result.method}")

    # ---------- power-scaling sensitivity ----------
    ps_result: dict[str, Any]
    try:
        ssm_model = builder._model
        observations = jnp.array(X.drop(columns=["time"]).values, dtype=jnp.float32)
        times = jnp.array(X["time"].values, dtype=jnp.float32)

        ps = power_scaling_sensitivity(
            model=ssm_model,
            observations=observations,
            times=times,
            result=result,
        )
        ps.print_report()
        ps_result = {
            "checked": True,
            "prior_sensitivity": ps.prior_sensitivity,
            "likelihood_sensitivity": ps.likelihood_sensitivity,
            "diagnosis": ps.diagnosis,
            "psis_k_hat": ps.psis_k_hat,
        }
    except Exception as e:
        print(f"Power-scaling check failed: {e}")
        ps_result = {"checked": False, "error": str(e)}

    # ---------- interventions ----------
    intervention_results: list[dict] = []
    try:
        samples = result.get_samples()
        spec = builder._spec
        latent_names = spec.latent_names or spec.manifest_names or []
        name_to_idx = {name: i for i, name in enumerate(latent_names)}

        outcome_idx = name_to_idx.get(outcome)
        drift_draws = samples.get("drift")
        cint_draws = samples.get("cint")

        if outcome_idx is not None and drift_draws is not None:
            n_latent = drift_draws.shape[-1]
            if cint_draws is None:
                cint_draws = jnp.zeros((drift_draws.shape[0], n_latent))

            id_status = causal_spec.get("identifiability") if causal_spec else None
            non_identifiable: set[str] = set()
            blocker_details: dict[str, list[str]] = {}
            if id_status:
                ni_map = id_status.get("non_identifiable_treatments", {})
                non_identifiable = set(ni_map.keys())
                blocker_details = {
                    t: d.get("confounders", []) for t, d in ni_map.items() if isinstance(d, dict)
                }

            for treatment_name in treatments:
                treat_idx = name_to_idx.get(treatment_name)
                if treat_idx is None:
                    intervention_results.append(
                        {
                            "treatment": treatment_name,
                            "effect_size": None,
                            "credible_interval": None,
                            "identifiable": treatment_name not in non_identifiable,
                            "warning": f"'{treatment_name}' not in latent model",
                        }
                    )
                    continue

                effects = vmap(
                    lambda d, c, ti=treat_idx, oi=outcome_idx: treatment_effect(d, c, ti, oi)
                )(drift_draws, cint_draws)

                mean_effect = float(jnp.mean(effects))
                q025 = float(jnp.percentile(effects, 2.5))
                q975 = float(jnp.percentile(effects, 97.5))
                prob_positive = float(jnp.mean(effects > 0))

                entry: dict[str, Any] = {
                    "treatment": treatment_name,
                    "effect_size": mean_effect,
                    "credible_interval": (q025, q975),
                    "prob_positive": prob_positive,
                    "identifiable": treatment_name not in non_identifiable,
                }
                if treatment_name in non_identifiable:
                    blockers = blocker_details.get(treatment_name, [])
                    if blockers:
                        entry["warning"] = (
                            f"Effect not identifiable (blocked by: {', '.join(blockers)})"
                        )
                    else:
                        entry["warning"] = "Effect not identifiable (missing proxies)"
                intervention_results.append(entry)

            intervention_results.sort(
                key=lambda x: abs(x["effect_size"]) if x["effect_size"] is not None else 0,
                reverse=True,
            )
    except Exception as e:
        print(f"Intervention analysis failed: {e}")
        intervention_results = [
            {
                "treatment": t,
                "effect_size": None,
                "credible_interval": None,
                "identifiable": True,
                "warning": str(e),
            }
            for t in treatments
        ]

    return {
        "ps_result": ps_result,
        "intervention_results": intervention_results,
    }


def run_stage5_gpu(
    stage4_result: dict,
    raw_data: pl.DataFrame,
    sampler_config: dict | None,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None,
    gpu: str,
) -> dict[str, Any]:
    """Dispatch stage 5 to a Modal GPU container.

    Serializes data as Arrow IPC bytes, sends everything to a remote
    Modal function, and returns plain-dict results.

    Args:
        stage4_result: Output from stage 4 (model_spec, priors, etc.)
        raw_data: Polars DataFrame with indicator/value/timestamp columns
        sampler_config: Sampler configuration dict (or None for defaults)
        treatments: List of treatment construct names
        outcome: Outcome variable name
        causal_spec: CausalSpec dict with identifiability status
        gpu: GPU type string (e.g. "A100", "L4", "B200")

    Returns:
        Dict with keys "ps_result" and "intervention_results"
    """
    import io

    import modal

    image = _make_image(gpu)
    app = modal.App("causal-ssm-stage5", image=image)

    # Register the remote function with GPU and timeout
    stage5_fn = app.function(gpu=gpu, timeout=7200)(_stage5_on_gpu)

    # Serialize DataFrame as Arrow IPC bytes
    buf = io.BytesIO()
    raw_data.write_ipc(buf)
    data_bytes = buf.getvalue()

    logger.info("Dispatching stage 5 to Modal (%s GPU)...", gpu)

    with app.run():
        return stage5_fn.remote(
            stage4_result=stage4_result,
            data_bytes=data_bytes,
            sampler_config=sampler_config,
            treatments=treatments,
            outcome=outcome,
            causal_spec=causal_spec,
        )
