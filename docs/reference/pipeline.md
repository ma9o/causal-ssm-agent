# Pipeline

## Overview

**[1a] Latent Model (Orchestrator):** Given only the user's question (no data), the Orchestrator LLM proposes a theoretical causal structure based on domain knowledge. It walks backwards from the implied outcome: what causes Y? What causes those? Output: latent constructs with causal edges. This separates theoretical reasoning from data-driven operationalization.

**[1b] Measurement Model (Orchestrator):** Given the latent structure and a data sample, the Orchestrator operationalizes each latent construct into observed indicators. For each latent, it proposes: `how_to_measure` instructions, `measurement_dtype`, `measurement_granularity`, and `aggregation`. One latent may map to multiple indicators (1:N reflective measurement model). Output: full DSEMStructure.

**[2] Distributed Discovery (Workers):** Worker LLMs process the full dataset in parallel chunks, extracting data for the proposed dimensions. (Future: workers may also critique the global graph based on local evidence, suggesting new confounders found only in specific chunks. The Orchestrator would then reconcile these structural suggestions into a unified model. Currently disabled—measurement extraction is the priority.)

**[3] Identifiability (y0):** Check if target causal effects are identifiable using Pearl's ID algorithm. In case of unobserved confounders that make effects unidentifiable, run sensitivity analysis (Cinelli-Hazlett) on a naive linear model and continue if bias is bounded (return control to user if not).

**[4] Model Specification (PyMC):** The orchestrator specifies the statistical model (GLMs) in PyMC and queries the workers for priors. (see: Zhu et al. 2024).

**[5] Inference:** Fit the model with PyMC, run proposed interventions and counterfactual simulations, return results to user ranked by effect size.

---

## Stage Files

Stages are in `src/dsem_agent/flows/stages/` with naming convention `stage{N}_{name}.py`.

---

## Implementation Notes

### Cross-Timescale Edge Aggregation (TODO: Functional Layer)

When implementing the functional layer, cross-timescale edges require aggregation:

- **Finer → Coarser** (e.g., hourly → daily): Aggregate the cause variable using its dimension's `aggregation` field. For example, if hourly `steps` (aggregation: "sum") affects daily `mood`, sum the 24 hourly step counts before modeling the relationship.

- **Coarser → Finer** (e.g., weekly → daily): No aggregation needed. The coarser variable's value is broadcast to all finer time points within its period.

The aggregation function is defined once per dimension (not per edge) because it captures the semantic meaning of how that variable should be rolled up (e.g., steps are summed, temperature is averaged).
