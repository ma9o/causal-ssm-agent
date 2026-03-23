# Stage 4b: Parametric Identifiability Diagnostics

Checks whether the chosen functional specification looks recoverable before full inference. It sits between Stage 4 and the inference backends described in [../model-runtime/inference-routing.md](../model-runtime/inference-routing.md), and corresponds to the pre-fit assurance surface described in [../concepts/pipeline-dimensions.md](../concepts/pipeline-dimensions.md).

## At a Glance

| Property | Value |
|---|---|
| Type | Computed |
| Interactive | No |
| Gate | Warning-only |
| Produces | [`ParametricIdResult`](../concepts/artifact-glossary.md) plus inference-structure summary |

## Inputs

| Input | Source | Description |
|---|---|---|
| `stage4.result` | Stage 4 | Model spec and priors, including the compiled SSM |
| `stage2.result` | Stage 2 | Model-ready data |

## Process

1. Run the T-rule counting screen.
2. Run output-sensitivity analysis for structural identifiability.
3. Run profile-likelihood analysis for practical identifiability.
4. Emit warnings when the model looks weakly identified or overparameterized.

## Outputs

| Output | Type | Description |
|---|---|---|
| `parametric_id` | `ParametricIdResult` | T-rule, sensitivity, and profile-likelihood results |
| `inference_structure` | `InferenceStructureResult?` | Active likelihood path and auto-routing summary |
| `gate_overridden` | `GateOverrideContract?` | Present if the warning gate was overridden |

## Key Structures

| Structure | Shape | Notes |
|---|---|---|
| `ParametricIdResult` | `{checked, t_rule, sensitivity_analysis?, summary, per_param_classification?, threshold?, error}` | Combined pre-fit diagnostic payload |
| `t_rule` | `{satisfies, n_free_params, n_moments}` | Necessary-condition check; the upstream model shape comes from [../model-runtime/functional-specification.md](../model-runtime/functional-specification.md) |
