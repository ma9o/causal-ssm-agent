# SSM Compilation Pipeline

The compilation pipeline translates a [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) (the Stage 4 output) into a NumPyro-ready `SSMModel`. The pipeline is a pure, deterministic transformation -- no LLM calls, no data access. It lives in `apps/data-pipeline/src/causal_ssm_agent/models/`.

Within the pipeline artifact lineage, this document sits between the [Stage 4 functional specification](../pipeline/04-model-specification-priors.md) and the estimation runtime. For the cross-cutting pipeline map, see [pipeline-dimensions.md](pipeline-dimensions.md).

## Data Flow

```text
[ModelSpec](../pipeline/04-model-specification-priors.md#modelspec) + [PriorProposal](../pipeline/04-model-specification-priors.md#priorproposal) + [CausalSpec](../pipeline/01b-measurement-identifiability.md#causalspec)
    │
    ▼
compile_ssm_artifact()                    [ssm_compiler.py]
    ├── validate_model_spec_for_compilation()
    │
    └── compile_ssm_inputs()              [ssm_compilation.py]
        ├── translate_spec()              [ssm_spec_translation.py]
        │   └── → SSMSpec + edge_lag_days
        │
        ├── compile_priors()              [ssm_prior_compilation.py]
        │   ├── build_prior_index_maps()  [ssm_prior_indexing.py]
        │   └── → SSMPriors + PriorIndexMaps
        │
        └── bind_parameters()             [ssm_prior_compilation.py]
            └── → parameter_bindings
    │
    ▼
CompiledSSMArtifact (serializable dict)
    │
    ▼
build_compiled_ssm_builder()              [ssm_builder.py]
    ├── deserialize_ssm_spec()
    ├── hydrate_discrete_manifest_metadata()  [ssm_observation_metadata.py]
    ├── validate_observation_support()
    └── → SSMModelBuilder
            │
            ▼
        builder.build_model(data) → SSMModel
        builder.fit(data) → InferenceResult
```

## Key Data Types

| Type | Defined in | Purpose |
|------|-----------|---------|
| [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) | `orchestrator/schemas_model.py` | User-facing model spec: parameters, likelihoods, roles |
| [`CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec) | `utils/causal_spec.py` | DAG edges, construct metadata, temporal granularity |
| `SSMSpec` | `models/ssm/model.py` | Structural SSM template: dimensions, masks, distributions |
| `SSMPriors` | `models/ssm/model.py` | Prior distributions for all SSM parameters |
| `PriorIndexMaps` | `ssm_compilation_common.py` | 5-tuple mapping param names → (prior field, flat index) |
| `CompiledSSMArtifact` | `ssm_compiler.py` | Serializable bundle: spec + priors + bindings |
| `SSMModel` | `models/ssm/model.py` | Executable NumPyro generative model |
| `InferenceResult` | `models/ssm/inference.py` | Posterior samples + diagnostics |

## Stage 1: Spec Translation (`ssm_spec_translation.py`)

Converts a `ModelSpec` + `CausalSpec` into an `SSMSpec` — the structural template defining dimensions and matrix masks.

**What it does:**

- Extracts latent construct layout from the DAG (names, order, time-invariant mask)
- Builds the **drift mask** (`n_latent x n_latent`): which AR and cross-lag entries are free (maps to the [CT-SDE drift matrix](estimation.md#1-ct-sde-formulation))
- Builds the **lambda matrix** (`n_manifest x n_latent`): indicator-to-construct mapping
- Computes **edge_lag_days**: for each cross-lag edge, the lag in days (used by prior compilation to scale DT→CT)
- Determines diffusion mode (`"diag"` or `"free"` if correlations are specified)
- Selects observation distribution families from `measurement_dtype`

**Key function:** `translate_spec(model_spec, causal_spec) -> (SSMSpec, edge_lag_days)`

## Stage 2: Prior Index Mapping (`ssm_prior_indexing.py`)

Builds the mapping from semantic parameter names (e.g., `"rho_mood"`, `"beta_mood_stress"`) to flat array indices in the SSM matrices.

**What it does:**

- For each parameter in `ModelSpec`, determines its role (AR coefficient, fixed effect, loading, residual SD, correlation)
- Maps it to the correct SSM field (`drift_diag`, `drift_offdiag`, `lambda_free`, etc.) and flat index
- Uses `split_compound_name()` to parse compound names like `"beta_mood_stress"` into (cause, effect)

**Key function:** `build_prior_index_maps(ssm_spec, model_spec, causal_spec) -> PriorIndexMaps`

**Returns:** 5-tuple of `dict[param_name -> (prior_field, flat_index)]`:

1. `offdiag_index` — cross-lag effects (drift off-diagonal)
2. `lambda_index` — factor loadings
3. `diag_index` — AR coefficients (drift diagonal)
4. `diffusion_diag_index` — residual SDs
5. `diffusion_offdiag_index` — residual correlations

## Stage 3: Prior Compilation (`ssm_prior_compilation.py`)

Translates user-facing prior specifications into `SSMPriors` arrays with the correct parameterization.

**Critical transformations:**

- **AR coefficients (DT→CT):** User specifies ρ ∈ (−1, 1) in discrete time. The compiler transforms to continuous-time drift diagonal via `μ_ct = −log(|ρ|)/dt`, `σ_ct = σ_ar / (μ_ar · dt)`.
- **Cross-lag effects:** Scaled by the reference interval (lag in days) for consistent CT interpretation.
- **Array assembly:** `build_array_prior_payload()` fills SSM-sized arrays from the sparse index maps, using defaults for unmapped slots.

**Key function:** `compile_priors(raw_priors, model_spec, ssm_spec, edge_lag_days, causal_spec) -> (SSMPriors, PriorIndexMaps)`

**Post-compilation checks:**

- `check_drift_lag_consistency()` — verifies DT-to-CT lag scaling is coherent
- `warn_first_order_approximation()` — flags when first-order Taylor expansion of DT→CT transform may be inaccurate

## Stage 4: Parameter Bindings (`ssm_prior_compilation.py`)

Creates the mapping from semantic parameter names to NumPyro sample sites — the bridge between the model specification and posterior extraction.

**Key function:** `bind_parameters(model_spec, ssm_spec, index_maps, causal_spec) -> list[dict]`

Each binding is: `{parameter: "rho_mood", site_name: "drift_diag_pop", flat_index: 0}`

This allows `InferenceResult` to map posterior samples back to user-facing parameter names.

## Stage 5: Artifact Serialization (`ssm_compiler.py`)

Bundles everything into a `CompiledSSMArtifact` — a JSON-serializable dict that can be persisted to disk and reconstructed later without re-running the compilation.

```python
CompiledSSMArtifact = {
    "schema_version": 1,
    "spec": {...},                      # SSMSpec as dict
    "compiled_prior_semantics": {...},   # PriorRuntimeBundle for deserialization
    "parameter_bindings": [...]          # Semantic name → NumPyro site mappings
}
```

Also provides validation entry points used by earlier pipeline stages:

- `validate_model_spec_for_compilation()` — catches structural errors before committing to compilation
- `trial_compile_model_spec()` / `trial_compile_measurement_model()` — dry-run compilation that returns an error string or None

## Stage 6: Builder & Runtime (`ssm_builder.py`, `ssm_observation_metadata.py`)

Reconstructs the compiled artifact into a live `SSMModelBuilder` that can build and fit models.

**`ssm_observation_metadata.py`** handles data-dependent hydration:

- `hydrate_discrete_manifest_metadata()` — infers level counts for categorical/ordinal emissions from observed data
- `validate_observation_support()` — checks no values fall outside the observation family's support (e.g., negative values for Poisson)

**`ssm_builder.py`** provides the runtime API:

- `build_model(data)` → `SSMModel` — materializes the NumPyro model with data
- `fit(data)` → `InferenceResult` — runs inference end-to-end
- `sample_prior_predictive()` — generates prior predictive samples for validation

## File Dependency Graph

```text
ssm_spec_translation.py ──┐
                           ├── ssm_compilation.py ──┬── ssm_compiler.py (public API)
ssm_prior_indexing.py ─────┤                        │
                           │                        └── ssm_builder.py (runtime API)
ssm_prior_compilation.py ──┘                             │
                                                         │
ssm_compilation_common.py ──── (shared by all above)     │
                                                         │
ssm_observation_metadata.py ─────────────────────────────┘
```

Leaf modules (`ssm_spec_translation`, `ssm_prior_indexing`, `ssm_observation_metadata`, `ssm_compilation_common`) have no intra-pipeline dependencies and can be understood in isolation. The compilation orchestrator (`ssm_compilation.py`) is the only file that calls all stages.
