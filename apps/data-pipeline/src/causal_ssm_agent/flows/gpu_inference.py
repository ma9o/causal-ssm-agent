"""Stage 5 GPU dispatch via Modal.

Runs fit_model + power_scaling + interventions together on a single
Modal GPU container so JAX arrays never cross the serialization boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import get_prefect_logger

if TYPE_CHECKING:
    import polars as pl

logger = get_prefect_logger(__name__)

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


def _stage5b_on_gpu(
    stage4_result: dict,
    data_bytes: bytes,
    sampler_config: dict | None,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None,
) -> dict[str, Any]:
    """Execute all stage 5b tasks inside a GPU container.

    This function runs *remotely* on Modal. All inputs/outputs are plain
    Python types (no JAX arrays cross the boundary).
    """
    import io

    import jax.numpy as jnp
    import polars as pl_inner

    from causal_ssm_agent.models.ssm.counterfactual import compute_interventions
    from causal_ssm_agent.models.ssm_builder import SSMModelBuilder
    from causal_ssm_agent.utils.parametric_id_postfit import power_scaling_sensitivity

    # ---------- reconstruct data ----------
    raw_data = pl_inner.read_ipc(io.BytesIO(data_bytes))

    # Verify float precision preserved through IPC (M10)
    for col_name, dtype in raw_data.schema.items():
        if dtype == pl_inner.Float32:
            raw_data = raw_data.with_columns(pl_inner.col(col_name).cast(pl_inner.Float64))

    # ---------- fit ----------
    model_spec = stage4_result.get("model_spec", {})
    priors = stage4_result.get("priors", {})
    cs = stage4_result.get("causal_spec")

    builder = SSMModelBuilder(
        model_spec=model_spec, priors=priors, sampler_config=sampler_config, causal_spec=cs
    )

    if raw_data.is_empty():
        return {
            "ps_result": {"checked": False, "error": "No data"},
            "intervention_results": [],
        }

    from causal_ssm_agent.utils.data import pivot_to_wide

    wide_data = pivot_to_wide(raw_data)

    result = builder.fit(wide_data)
    logger.info("Fit complete: method=%s", result.method)

    # Prepare shared arrays used by multiple diagnostics
    manifest_cols = [c for c in wide_data.columns if c != "time"]
    observations = jnp.array(wide_data.select(manifest_cols).to_numpy(), dtype=jnp.float32)
    times = jnp.array(wide_data["time"].to_numpy(), dtype=jnp.float32)

    # Extract serializable diagnostics before they get lost across the boundary
    mcmc_diag = result.get_mcmc_diagnostics()
    svi_diag = result.get_svi_diagnostics()

    # LOO diagnostics
    import functools

    assert builder._model is not None
    model_fn = functools.partial(
        builder._model.model,
        likelihood_backend=builder._model.make_likelihood_backend(),
    )
    loo_diag = result.get_loo_diagnostics(
        model_fn=model_fn,
        observations=observations,
        times=times,
    )

    # Posterior marginals and pairs
    posterior_marginals = result.get_posterior_marginals()
    posterior_pairs = result.get_posterior_pairs()

    # ---------- power-scaling sensitivity ----------
    ps_result: dict[str, Any]
    try:
        assert builder._model is not None
        ssm_model = builder._model

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
    except Exception:
        logger.exception("Power-scaling check failed")
        ps_result = {"checked": False, "error": "see logs for traceback"}

    # ---------- posterior predictive checks ----------
    ppc_result: dict[str, Any]
    try:
        from causal_ssm_agent.models.posterior_predictive import (
            run_posterior_predictive_checks,
        )

        assert builder._model is not None
        spec = builder._spec
        assert spec is not None
        manifest_names = spec.manifest_names or manifest_cols
        manifest_dist_val = (
            spec.manifest_dist.value
            if hasattr(spec.manifest_dist, "value")
            else str(spec.manifest_dist)
        )

        # Per-channel distributions override scalar fallback
        manifest_dists_list = None
        if spec.manifest_dists:
            manifest_dists_list = [
                d.value if hasattr(d, "value") else str(d) for d in spec.manifest_dists
            ]

        ppc = run_posterior_predictive_checks(
            samples=result.get_samples(),
            observations=observations,
            times=times,
            manifest_names=manifest_names,
            manifest_dist=manifest_dist_val,
            manifest_dists=manifest_dists_list,
        )
        ppc_result = ppc.model_dump(mode="json")
    except Exception:
        logger.exception("PPC check failed")
        ppc_result = {"checked": False, "error": "see logs for traceback"}

    # ---------- interventions ----------
    try:
        samples = result.get_samples()
        spec = builder._spec
        assert spec is not None
        latent_names = spec.latent_names
        if latent_names is None:
            logger.warning(
                "SSMSpec.latent_names is None; falling back to manifest_names"
                " — intervention indices may be incorrect"
            )
            latent_names = spec.manifest_names or []

        intervention_results = compute_interventions(
            samples=samples,
            treatments=treatments,
            outcome=outcome,
            latent_names=latent_names,
            causal_spec=causal_spec,
            ppc_result=ppc_result,
            manifest_names=spec.manifest_names or [],
            ps_result=ps_result,
            times=times,
        )
    except Exception as e:
        logger.exception("Intervention analysis failed")
        intervention_results = [
            {
                "treatment": t,
                "effect_size": None,
                "identifiable": True,
                "warning": str(e),
            }
            for t in treatments
        ]

    return {
        "ps_result": ps_result,
        "ppc_result": ppc_result,
        "intervention_results": intervention_results,
        "mcmc_diagnostics": mcmc_diag,
        "svi_diagnostics": svi_diag,
        "loo_diagnostics": loo_diag,
        "posterior_marginals": posterior_marginals,
        "posterior_pairs": posterior_pairs,
    }


def run_stage5b_gpu(
    stage4_result: dict,
    raw_data: pl.DataFrame,
    sampler_config: dict | None,
    treatments: list[str],
    outcome: str,
    causal_spec: dict | None,
    gpu: str,
) -> dict[str, Any]:
    """Dispatch stage 5b to a Modal GPU container.

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
        Dict with keys "ps_result", "ppc_result", and "intervention_results"
    """
    import io

    import modal

    image = _make_image(gpu)
    app = modal.App("causal-ssm-stage5b", image=image)

    # Register the remote function with GPU and timeout
    stage5b_fn = app.function(gpu=gpu, timeout=7200)(_stage5b_on_gpu)

    # Serialize DataFrame as Arrow IPC bytes
    buf = io.BytesIO()
    raw_data.write_ipc(buf)
    data_bytes = buf.getvalue()

    logger.info("Dispatching stage 5b to Modal (%s GPU)...", gpu)

    with app.run():
        return stage5b_fn.remote(
            stage4_result=stage4_result,
            data_bytes=data_bytes,
            sampler_config=sampler_config,
            treatments=treatments,
            outcome=outcome,
            causal_spec=causal_spec,
        )
