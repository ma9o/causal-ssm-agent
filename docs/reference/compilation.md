# SSM Compilation Pipeline

The compilation pipeline translates a [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) (the Stage 4 output) into a NumPyro-ready `SSMModel`. The resulting `CompiledSSMArtifact` is consumed by [Stage 5b](../pipeline/05b-inference-diagnostics.md) for fitting.

```mermaid
graph TD
    ModelSpec(["ModelSpec"])
    PriorProposal(["PriorProposal"])
    CausalSpec(["CausalSpec"])

    ModelSpec & PriorProposal & CausalSpec --> validate

    subgraph compile_ssm_artifact ["compile_ssm_artifact() — ssm_compiler.py"]
        validate["validate_model_spec_for_compilation()"]
        validate --> translate

        subgraph compile_model_spec ["compile_ssm_inputs_from_model_spec() — ssm_compilation.py"]
            translate["translate_spec() — ssm_spec_translation.py"]
            translate --> translate_out(["SSMSpec + edge_lag_days"])
            translate_out --> prior_idx["build_prior_index_maps() — ssm_prior_indexing.py"]
            prior_idx --> priors["compile_priors() — ssm_prior_compilation.py"]
            translate_out --> priors
            priors --> priors_out(["PriorRegistry + PriorIndexMaps"])
            priors_out --> bind["bind_parameters() — ssm_prior_compilation.py"]
            bind --> bind_out(["parameter_bindings"])
            priors_out --> attach_diag["_attach_compile_binding_provenance()"]
            bind_out --> attach_diag
            attach_diag --> diag_out(["compile_diagnostics"])
        end

        priors_out --> prior_sem["compile_prior_semantics() — parameterization.py"]
    end

    translate_out --> artifact
    prior_sem --> artifact
    bind_out --> artifact
    diag_out --> artifact
    artifact["CompiledSSMArtifact\nspec + edge_lag_days + compiled_prior_semantics + parameter_bindings + compile_diagnostics"]

    subgraph prepare_model_runtime ["prepare_model_runtime() — runtime.py"]
        artifact --> model_ctor["build_model_from_compiled_artifact()\n(deserialize_ssm_spec + load_prior_runtime_bundle)"]
        model_ctor --> hydrate["hydrate_discrete_manifest_metadata() — observation_support.py"]
        hydrate --> validate_obs["validate_observation_support()"]
        validate_obs --> ssm_model(["SSMModel"])
        ssm_model --> arrays["prepare_fit_inputs()\nobservations + times + manifest order"]
        arrays --> support["compile_observation_support_runtime()"]
        support --> registry["Site registry + PriorRuntimeBundle\n(derived from SSMSpec)"]
        registry --> assemble["Runtime assembly\n(component params + block params -> RuntimeDynamics / diffusion / loadings / t0)"]
    end

    ssm_model --> fit["fit_prepared_model(runtime) → InferenceResult"]
    fit --> execute["SSMModel.model() execution"]
    assemble --> execute

    click ModelSpec "../pipeline/04-model-specification-priors.md#modelspec"
    click PriorProposal "../pipeline/04-model-specification-priors.md#priorproposal"
    click CausalSpec "../pipeline/01b-measurement-identifiability.md#causalspec"
```

## Key Data Types

| Type | Defined in | Purpose |
|------|-----------|---------|
| [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) | `artifacts/model_spec.py` | User-facing model spec: parameters, likelihoods, roles |
| [`CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec) | `artifacts/causal_spec.py` | DAG edges, construct metadata, temporal granularity |
| `SSMSpec` | `models/ssm/model.py` | SSM artifact: dimensions, names, distributions, structure blocks, and composite drift spec |
| `SiteDescriptor` | `models/ssm/structure/sites.py` | Canonical sample-site identity: name, shape, support, semantic kind, assembly group, and prior binding field |
| `PriorRegistry` | `models/ssm/priors.py` | Site-keyed canonical priors for both structure blocks and dynamics components |
| `PriorRuntimeBundle` | `models/ssm/parameterization.py` | Runtime site registry, transforms, and prior-state arrays reconstructed without model tracing |
| `PriorIndexMaps` | `models/ssm/compile/common.py` | 14-tuple mapping param names → (prior field, flat index) for semantic bindings |
| `CompiledSSMArtifact` | `models/ssm/compile/artifact.py` | Serializable bundle: spec + edge lags + compiled prior semantics + bindings + diagnostics |
| `SSMModel` | `models/ssm/model.py` | Executable NumPyro generative model |
| `InferenceResult` | `models/ssm/inference/types.py` | Posterior samples + diagnostics |

## Stage 1: Spec Translation (`ssm_spec_translation.py`)

Converts a `ModelSpec` + `CausalSpec` into an `SSMSpec` — the artifact that persists concrete numeric templates, structure blocks, and the composite drift spec.

**What it does:**

- Extracts latent construct layout from the DAG (names, order, time-invariant mask)
- Builds the **dynamics spec** as a composite vector field. The standard affine artifact uses `DiagonalDecaySpec` for per-latent decay, `LinearEdgeSpec` for cross-lag dynamics, and `StateInterceptSpec` for any free continuous-time state intercepts.
- Builds the **loading template** (`lambda_mat`) plus `lambda_mask`: fixed indicator-to-construct loadings and free non-reference loadings
- Compiles concrete templates plus masks for `cint`, `static_state_sds`, `diffusion_chol`, `manifest_means`, `manifest_chol`, `t0_means`, and `t0_chol`.
- Converts marginalized time-invariant confounders into compiled low-rank baseline factors of the form `B diag(tau^2) B^T` rather than free pairwise `cor0_*` surfaces on the causal-spec path
- Derives deterministic `manifest_centered` flags from the locked likelihood family, link, and observation support semantics
- Applies `initialization_policy` and `equilibrium_forcing` to determine which `t0_*`, `cint_*`, and `manifest_mean_*` surfaces remain free
- Computes **edge_lag_days**: for each cross-lag edge, the lag in days (used by prior compilation to scale DT→CT)
- Selects observation distribution families from `measurement_dtype`
- Eliminates legacy string matrix modes: translation always emits concrete templates and explicit masks

**Key function:** `translate_spec(model_spec, causal_spec) -> (SSMSpec, edge_lag_days)`

## Stage 2: Prior Index Mapping (`ssm_prior_indexing.py`)

Builds the mapping from semantic parameter names (e.g., `"rho_mood"`, `"beta_mood_stress"`) to flat array indices in the SSM matrices.

**What it does:**

- For each parameter in `ModelSpec`, determines its role (AR coefficient, fixed effect, loading, residual SD, state intercept, observation intercept, static baseline-factor SD, correlation, and observation-family auxiliary site)
- Maps it to the correct SSM field (`drift_base_decay`, `drift_offdiag`, `lambda_free`, `manifest_means`, `cint`, `static_state_sd`, etc.) and flat index
- Derives the canonical free-entry order from block and dynamics `SiteDescriptor`s so the compiler, runtime assembly, and posterior name resolution all share one site registry.
- Uses `split_compound_name()` to parse compound names like `"beta_mood_stress"` into (cause, effect)

**Key function:** `build_prior_index_maps(ssm_spec, model_spec, causal_spec) -> PriorIndexMaps`

This is now a strict internal helper: it requires both a translated `SSMSpec` and a semantic `ModelSpec`. Spec-only entrypoints decide explicitly when no semantic bindings should be produced; the indexer no longer falls back to all-empty maps.

**Returns:** 14-tuple of `dict[param_name -> (prior_field, flat_index)]`:

1. `offdiag_index` — cross-lag effects (drift off-diagonal)
2. `lambda_index` — factor loadings
3. `diag_index` — AR coefficients (baseline decay for the derived drift diagonal)
4. `diffusion_diag_index` — residual SDs
5. `diffusion_offdiag_index` — residual correlations
6. `input_effect_index` — transition input-effect entries
7. `t0_offdiag_index` — initial-state correlations
8. `t0_mean_index` — initial-state means
9. `t0_sd_index` — initial-state standard deviations
10. `manifest_mean_index` — manifest intercepts
11. `manifest_var_index` — manifest noise terms
12. `cint_index` — continuous-time state intercepts
13. `static_state_sd_index` — compiled baseline-factor scales
14. `observation_site_index` — observation-family hyperparameter sites

## Stage 3: Prior Compilation (`ssm_prior_compilation.py`)

Translates user-facing prior specifications into a site-keyed `PriorRegistry` with the correct parameterization.

**Critical transformations:**

- **AR coefficients (DT→CT):** User specifies `rho_*` as baseline persistence in `(0, 1)` over the authored interval, absent incoming feedback. The compiler transforms it to positive continuous-time base decay with `base_decay = −log(rho) / dt`; nondegenerate priors are moment-matched to `Gamma(concentration, rate)`, and fixed-width `rho_*` priors compile to positive `Delta(value)`.
- **Hard-sparsity drift assembly:** Off-diagonal entries are compiled as `A_ij = beta_ij / dt` on allowed edges only. For each dynamic row, the realised diagonal is derived as `A_ii = -(base_decay_i + sum_j |A_ij| + stability_margin)`, preserving structural zeros while guaranteeing strict row diagonal dominance.
- **Cross-lag effects:** Scaled by an explicitly resolved positive interval in this order: `reference_interval_days`, then compiled `edge_lag_days`, then the causal-spec model clock. If none exists, compilation now raises instead of silently assuming `1.0d`.
- **Site binding:** compiled priors attach to canonical `SiteDescriptor`s. Structure blocks and dynamics components use the same prior materialization path.

**Key function:** `compile_priors(raw_priors, model_spec, ssm_spec, edge_lag_days, causal_spec) -> (PriorRegistry, PriorIndexMaps, list[CompileDiagnostic])`

**Post-compilation diagnostics:**

- `collect_interval_provenance_warnings()` — warns when cited source intervals and authored/model intervals materially disagree
- `collect_first_order_approximation_warnings()` — uses the full matrix logarithm `logm(A_dt) / dt` to flag cross-lag priors whose elementwise `beta_dt / dt` CT coupling materially differs from the full-system CT scale
- `collect_compile_diagnostics()` — combines these warnings into the structured diagnostics payload persisted on the artifact

## Stage 4: Parameter Bindings (`ssm_prior_compilation.py`)

Creates the mapping from semantic parameter names to NumPyro sample sites — the bridge between the model specification and posterior extraction.

**Key function:** `bind_parameters(index_maps) -> list[dict]`

Each binding is: `{parameter: "rho_mood", site_name: "vf_0_base_decay", flat_index: 0}`

This allows `InferenceResult` to map posterior samples back to user-facing parameter names. `bind_parameters()` consumes the already-compiled `PriorIndexMaps` from Stage 3 rather than rebuilding them, and only the compile entrypoints decide whether semantic bindings should exist at all.

## Stage 5: Artifact Serialization (`ssm_compiler.py`)

Bundles everything into a `CompiledSSMArtifact` — a JSON-serializable dict that can be persisted to disk and reconstructed later without re-running compilation.

```python
CompiledSSMArtifact = {
    "schema_version": 1,
    "spec": {...},                      # SSMSpec as dict
    "edge_lag_days": [...],             # serialized lag metadata per retained edge
    "compiled_prior_semantics": {...},  # serialized PriorRuntimeBundle payload
    "parameter_bindings": [...],        # semantic name → NumPyro site mappings
    "compile_diagnostics": [...],       # compile-time warnings / notes
}
```

The artifact stores `compiled_prior_semantics`, the canonical runtime prior block used to reconstruct `PriorRuntimeBundle` without retracing the model.

Also provides validation entry points used by earlier pipeline stages:

- `validate_model_spec_for_compilation()` — catches structural errors before committing to compilation
- `trial_compile_model_spec()` / `trial_compile_measurement_model()` — dry-run compilation that returns an error string or None

## Stage 5: Runtime Preparation (`compile/artifact.py`, `runtime.py`, `observation_support.py`)

`build_model_from_compiled_artifact()` in `compile/artifact.py` reconstructs the compiled artifact into a live `SSMModel`.

**`observation_support.py`** handles data-dependent hydration:

- `hydrate_discrete_manifest_metadata()` — infers level counts for categorical/ordinal emissions from observed data
- `validate_observation_support()` — checks no values fall outside the observation family's support (e.g., negative values for Poisson)

**`runtime.py`** provides the runtime API:

- `build_ssm_model(wide_data, ...)` → `SSMModel` — materializes the NumPyro model from direct specs or compiled inputs
- `prepare_model_runtime(data_for_model, ...)` → `PreparedModelRuntime` — prepares observations, times, observation support, transition inputs, and sampler config
- `fit_prepared_model(runtime)` → `InferenceResult` — routes prepared arrays into `inference.fit()`
- `sample_prior_predictive(model, ...)` — generates prior predictive samples for validation

Runtime reconstruction now has three layers:

- **`SSMSpec`** remains the serialization and validation boundary. It owns persisted templates, masks, distributions, names, and the composite drift spec.
- **`PriorRuntimeBundle`** is rebuilt from `compiled_prior_semantics` and owns sample-site registry, transforms, and prior-state semantics.
- **`RuntimeDynamics`** is sampled through the compiled composite drift spec. Inference backends then derive affine or local-linear views from the vector field instead of splitting linear vs nonlinear at the spec level.

**Why `fit()` stays outside `SSMModel`:** the runtime separates three concerns:

- **`SSMModel`** is a pure NumPyro model function. Its [`model(observations, times)`](estimation.md#data-flow) method takes JAX arrays, samples from the runtime prior bundle, assembles block deterministic values plus `RuntimeDynamics`, and injects the log-likelihood via `numpyro.factor()`. It has no knowledge of DataFrames, inference algorithms, or sampler configuration.
- **`inference.fit()`** handles [algorithm selection](inference-routing.md) and execution. It takes an `SSMModel` and prepared arrays.
- **`PreparedModelRuntime`** bridges the gap: it carries the `SSMModel`, prepared JAX arrays, observation-support runtime, transition inputs, manifest order, sampler config, and inference-structure plan for Stage 5 diagnostics.
- Before array conversion, `prepare_fit_inputs()` applies deterministic centering to manifest columns whose compiled `manifest_centered` flag is `True`, so centered additive-location indicators are zero-centered consistently in both fitting and prior-predictive scale checks.

Moving `fit()` onto `SSMModel` would couple it to DataFrame handling and sampler config routing — concerns that belong to the orchestrator layer, not the probabilistic model.

**Two model-construction entry points:**

- `build_model_from_compiled_artifact(compiled_ssm, wide_data)` in `compile/artifact.py` — deserializes a persisted `CompiledSSMArtifact`, rebuilds the prior runtime bundle from `compiled_prior_semantics`, hydrates observation metadata from wide data, and returns an `SSMModel`. This is the pipeline path because Stage 5 consumes the artifact that Stage 4 persisted.
- `build_ssm_model(wide_data, model_spec=..., priors=..., causal_spec=...)` in `runtime.py` — compiles on the fly from raw specs and returns an `SSMModel`. This is the direct path for tests and notebooks.

Both return a live `SSMModel`. Callers that need fit-ready arrays use `prepare_model_runtime()` or `prepare_wide_model_runtime()` to construct a `PreparedModelRuntime`.

## File Dependency Graph

```mermaid
graph LR
    spec["compile/spec_translation.py"] --> inputs["compile/inputs.py"]
    indexing["compile/prior_indexing.py"] --> inputs
    prior["compile/prior_compilation.py"] --> inputs
    inputs --> artifact["compile/artifact.py"]
    inputs --> runtime["runtime.py"]
    artifact --> runtime
    obs["observation_support.py"] --> runtime
    common["compile/common.py"] -.-> spec & indexing & prior & inputs & artifact & runtime
```

Leaf modules (`compile/spec_translation.py`, `compile/prior_indexing.py`, `observation_support.py`, `compile/common.py`) have no intra-pipeline dependencies and can be understood in isolation. The compilation orchestrator (`compile/inputs.py`) exposes two explicit entry points:

- `compile_ssm_inputs_from_model_spec()` for the semantic Stage 4 path
- `compile_ssm_inputs_from_spec()` for already-translated `SSMSpec` callers
