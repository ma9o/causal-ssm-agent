# Documentation Index

## Start Here

| Need | Open |
|---|---|
| Artifact pipeline walkthrough | [pipeline.md](pipeline.md) |
| Contributor and operator workflows | [guides/](guides/.md) |
| Cross-cutting runtime and modeling references | [reference/](reference/) |

## Artifact Owners

| Artifact or concept | Owning doc |
|---|---|
| `LatentStructure` | [pipeline/latent-structure.md](pipeline/latent-structure.md) |
| `MeasurementStructure`, `Indicator`, `CausalDesign`, `IdentifiabilityStatus` | [pipeline/measurement-structure.md](pipeline/measurement-structure.md) |
| `ObservationRecord`s and the encoded observation table (`data_for_model`) | [pipeline/extraction.md](pipeline/extraction.md) |
| `IndicatorAudit` and validation findings | [pipeline/extraction-validation.md](pipeline/extraction-validation.md) |
| `StatisticalModelSpec`, `LikelihoodSpec`, `ParameterSpec`, `PriorProposal` | [pipeline/statistical-model-spec.md](pipeline/statistical-model-spec.md) |
| `FittedArtifact` and post-fit diagnostics | [pipeline/inference.md](pipeline/inference.md) |
| `TreatmentEffect` and intervention tools | [pipeline/analysis.md](pipeline/analysis.md) |

## Cross-Cutting References

| Topic | Open |
|---|---|
| Artifact lineage, temporal layers, assurance surfaces | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md) |
| Temporal workflow/activity structure for LLM transitions | [design/temporal-llm-orchestration.md](design/temporal-llm-orchestration.md) |
| Compilation from `statistical_model_spec` outputs to executable SSM runtime | [reference/compilation.md](reference/compilation.md) |
| Continuous-time estimation and discretization | [reference/estimation.md](reference/estimation.md) |
| Inference-method selection and structural routing | [reference/inference-routing.md](reference/inference-routing.md) |
| High-level map of the model-spec reducer and repair loop | [reference/statistical-model-spec/state-machine.md](reference/statistical-model-spec/state-machine.md) |
| How model-spec constrains LLM model-form and prior decisions | [reference/statistical-model-spec/llm-driven-specification.md](reference/statistical-model-spec/llm-driven-specification.md) |

## Guides

| Workflow | Open |
|---|---|
| Local setup | [Guide](guides/dev_setup.md) |
| TypeScript code generation from Python contracts | [Guide](guides/codegen.md) |
| Integration testing | [Guide](guides/agentic_integration_testing.md) |
| Evaluations | [Guide](guides/running_evals.md) |

## Route by Question

| Question | Open first |
|---|---|
| What does this transition emit? | The relevant file under `pipeline/` |
| What does this artifact mean? | The artifact doc that introduces it |
| What assumptions constrain that artifact? | The matching page under `reference/` |
| How does time work across extraction, fitting, and interventions? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md) |
