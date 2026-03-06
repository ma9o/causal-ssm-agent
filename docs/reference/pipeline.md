# Pipeline

## Overview

**[1a] Latent Model (Orchestrator):** Given only the user's question (no data), the Orchestrator LLM proposes a theoretical causal structure based on domain knowledge. It walks backwards from the implied outcome: what causes Y? What causes those? Output: latent constructs with causal edges. This separates theoretical reasoning from data-driven operationalization.

**[1b] Measurement Model with Identifiability (Orchestrator):** Given the latent structure and a data sample, the Orchestrator operationalizes each latent construct into observed indicators. For each latent, it proposes: `how_to_measure` instructions, `measurement_dtype`, and `aggregation`. Temporal resolution is controlled by a pipeline-level `aggregation_window` rather than per-indicator granularity. One latent may map to multiple indicators (1:N reflective measurement model).

After proposing measurements, identifiability is checked using y0's ID algorithm (Pearl's do-calculus). If effects are non-identifiable due to unobserved confounders, the Orchestrator is prompted to propose proxy indicators for the blocking confounders. Identifiability is re-checked after adding proxies. Effects that remain non-identifiable are flagged in the model for downstream handling. Output: full CausalSpec with identifiability status.

**[2] Extract (Workers):** Worker LLMs process the full dataset in parallel chunks, extracting raw indicator values as (indicator, value, timestamp) tuples. This is the "E" in ETL. Workers do one thing: extract values from text. (Future: workers may also critique the global graph based on local evidence, suggesting new confounders found only in specific chunks. The Orchestrator would then reconcile these structural suggestions into a unified model. Currently disabled—measurement extraction is the priority.)

**[3] Transform + Validate:** This stage performs the "T" (Transform) and semantic validation in an ETL pipeline:

**Stage 3a - Transform (aggregate_measurements):**
- Concatenate raw worker DataFrames (indicator, value, timestamp)
- Parse timestamps and bucket to each construct's temporal_scale
- Apply indicator-specific aggregation (mean, sum, max, etc.)
- Output: dict[granularity → DataFrame] with time_bucket column + indicator columns

**Stage 3b - Validate (validate_extraction):**
Semantic checks that Polars schema can't enforce. Structural validation (column existence, dtypes) is handled by Polars and downstream stages.

| Check | What | Failure Condition |
|-------|------|-------------------|
| **Variance** | Indicator has variance > 0 | Constant values (zero information) |
| **Sample size** | Enough time points for temporal modeling | < N observations per granularity |

**Output:** `{is_valid: bool, issues: list[{indicator, issue_type, severity, message}]}`

**[4] Model Specification (NumPyro/JAX):** The orchestrator specifies the Bayesian hierarchical state-space model and queries the workers for priors. The output is consumed by `SSMModelBuilder` to produce an `SSMSpec`/`SSMModel`. (see: Zhu et al. 2024).

**[4b] Parametric Identifiability:** Pre-fit diagnostics that check whether model parameters are constrained by the data before running expensive inference. Detects structural non-identifiability (rank-deficient Fisher information), boundary identifiability, and weak parameters. See `src/causal_ssm_agent/flows/stages/stage4b_parametric_id.py`.

**[5] Inference:** Fit the model with NumPyro/JAX (SVI, NUTS, NUTS-DA, Hess-MC2, PGAS, Tempered SMC, Laplace-EM, Structured VI, or DPF), run proposed interventions, return results to user ranked by effect size.

---

## Stage Files

Stages are in `src/causal_ssm_agent/flows/stages/` with naming convention `stage{N}_{name}.py`.

---

## Implementation Notes

### Cross-Timescale Edge Aggregation (TODO: Functional Layer)

When implementing the functional layer, cross-timescale edges require aggregation:

- **Finer → Coarser** (e.g., hourly → daily): Aggregate the cause variable using its dimension's `aggregation` field. For example, if hourly `steps` (aggregation: "sum") affects daily `mood`, sum the 24 hourly step counts before modeling the relationship.

- **Coarser → Finer** (e.g., weekly → daily): No aggregation needed. The coarser variable's value is broadcast to all finer time points within its period.

The aggregation function is defined once per dimension (not per edge) because it captures the semantic meaning of how that variable should be rolled up (e.g., steps are summed, temperature is averaged).
