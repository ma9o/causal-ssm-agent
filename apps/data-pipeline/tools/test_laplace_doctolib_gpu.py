"""Run the Doctolib laplace_em smoke test on a GPU via Modal.

Usage:
    modal run tools/test_laplace_doctolib_gpu.py
    modal run tools/test_laplace_doctolib_gpu.py --gpu A100
"""

from pathlib import Path

import modal

ROOT = Path(__file__).parent.parent  # apps/data-pipeline
FIXTURE_DIR = ROOT.parent.parent / "data" / "DOCTOLIB" / "run"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv")
    .uv_sync(uv_project_dir=str(ROOT), groups=["dev", "cloud"], frozen=True)
    .uv_pip_install("jax[cuda12]", gpu="A100")
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(ROOT / "src" / "causal_ssm_agent", remote_path="/root/src/causal_ssm_agent")
    .add_local_dir(FIXTURE_DIR, remote_path="/root/fixtures/doctolib")
    .add_local_file(ROOT / "config.yaml", remote_path="/root/src/config.yaml")
)

app = modal.App("causal-ssm-laplace-doctolib", image=image)


@app.function(gpu="A100", timeout=1800)
def run_laplace_doctolib():
    import json
    import time
    import traceback
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import polars as pl

    print(f"JAX {jax.__version__}  backend={jax.default_backend()}  devices={jax.devices()}")

    fixture_dir = Path("/root/fixtures/doctolib")

    def load(name):
        return json.loads((fixture_dir / name).read_text())

    stage4 = load("stage-4.json")
    stage1b = load("stage-1b.json")["causal_spec"]
    raw_data = pl.read_parquet(fixture_dir / "stage2-raw-data.parquet")

    name_map = {
        "beta_lipid_cv": "beta_lipid_burden_cardiovascular_risk",
        "beta_pressure_cv": "beta_arterial_pressure_cardiovascular_risk",
        "beta_glycemic_cv": "beta_glycemic_control_cardiovascular_risk",
        "beta_lipid_inflammation": "beta_lipid_burden_vascular_inflammation",
        "beta_inflammation_cv": "beta_vascular_inflammation_cardiovascular_risk",
        "beta_adherence_lipid": "beta_medication_adherence_lipid_burden",
        "beta_adherence_pressure": "beta_medication_adherence_arterial_pressure",
        "rho_lipid": "rho_lipid_burden",
        "rho_pressure": "rho_arterial_pressure",
        "sigma_lipid": "sigma_lipid_burden",
        "sigma_pressure": "sigma_arterial_pressure",
        "rho_inflammation": "rho_vascular_inflammation",
    }

    model_spec = json.loads(json.dumps(stage4["model_spec"]))
    for parameter in model_spec["parameters"]:
        parameter["name"] = name_map.get(parameter["name"], parameter["name"])
        if parameter["role"] == "ar_coefficient":
            parameter["constraint"] = "correlation"

    priors = json.loads(json.dumps(stage4["priors"]))
    for old_name, new_name in name_map.items():
        if old_name in priors:
            priors[new_name] = priors.pop(old_name)

    stage4_construct_names = {
        "medication_adherence",
        "lipid_burden",
        "vascular_inflammation",
        "glycemic_control",
        "arterial_pressure",
        "cardiovascular_risk",
    }
    measurement = {
        "indicators": [
            ind
            for ind in stage1b["measurement"]["indicators"]
            if ind["construct_name"] in stage4_construct_names
        ]
    }
    causal_spec = {
        "latent": {
            "constructs": [
                {
                    "name": "medication_adherence",
                    "description": "Prescription refill.",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "monthly",
                },
                {
                    "name": "lipid_burden",
                    "description": "Atherogenic lipid profile.",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "monthly",
                },
                {
                    "name": "vascular_inflammation",
                    "description": "Inflammatory state.",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "monthly",
                },
                {
                    "name": "glycemic_control",
                    "description": "Blood-glucose regulation.",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "monthly",
                },
                {
                    "name": "arterial_pressure",
                    "description": "Blood-pressure burden.",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "temporal_scale": "monthly",
                },
                {
                    "name": "cardiovascular_risk",
                    "description": "Cardiovascular risk.",
                    "role": "endogenous",
                    "is_outcome": True,
                    "temporal_status": "time_varying",
                    "temporal_scale": "monthly",
                },
            ],
            "edges": [
                {
                    "cause": "medication_adherence",
                    "effect": "lipid_burden",
                    "description": "Adherence improves lipid control.",
                },
                {
                    "cause": "medication_adherence",
                    "effect": "arterial_pressure",
                    "description": "Adherence improves BP.",
                },
                {
                    "cause": "lipid_burden",
                    "effect": "vascular_inflammation",
                    "description": "Lipids raise inflammation.",
                },
                {
                    "cause": "lipid_burden",
                    "effect": "cardiovascular_risk",
                    "description": "Lipids raise CV risk.",
                },
                {
                    "cause": "vascular_inflammation",
                    "effect": "cardiovascular_risk",
                    "description": "Inflammation raises CV risk.",
                },
                {
                    "cause": "glycemic_control",
                    "effect": "cardiovascular_risk",
                    "description": "Poor glycemia raises CV risk.",
                },
                {
                    "cause": "arterial_pressure",
                    "effect": "cardiovascular_risk",
                    "description": "BP raises CV risk.",
                },
            ],
        },
        "measurement": measurement,
    }

    from causal_ssm_agent.models.ssm import InferenceResult
    from causal_ssm_agent.models.ssm_builder import build_ssm_builder
    from causal_ssm_agent.models.ssm_compiler import compile_ssm_artifact
    from causal_ssm_agent.utils.data import pivot_to_wide

    compiled = compile_ssm_artifact(model_spec, priors, causal_spec=causal_spec)
    builder = build_ssm_builder(
        raw_data=raw_data,
        compiled_ssm=compiled,
        sampler_config={
            "method": "laplace_em",
            "n_outer": 6,
            "n_csmc_particles": 8,
            "n_mh_steps": 3,
            "param_step_size": 0.05,
            "n_warmup": 3,
            "n_ieks_iters": 3,
            "adaptive_tempering": False,
            "seed": 0,
        },
    )

    wide = pivot_to_wide(raw_data)
    print(f"Data shape: {wide.shape}, manifest_dist: {builder._spec.manifest_dist}")
    print(f"manifest_dists: {builder._spec.manifest_dists}")

    t0 = time.perf_counter()
    try:
        result = builder.fit(wide)
        elapsed = time.perf_counter() - t0
        assert isinstance(result, InferenceResult)
        samples = result.get_samples()
        print(f"\nSUCCESS in {elapsed:.1f}s")
        print(f"  method={result.method}")
        print(f"  drift_diag_pop shape: {samples['drift_diag_pop'].shape}")
        print(f"  finite: {bool(jnp.isfinite(samples['drift_diag_pop']).all())}")
        print(f"  accept_rates: {result.diagnostics.get('accept_rates', [])}")
    except Exception:
        elapsed = time.perf_counter() - t0
        print(f"\nFAILED after {elapsed:.1f}s")
        traceback.print_exc()


@app.local_entrypoint()
def main():
    run_laplace_doctolib.remote()
