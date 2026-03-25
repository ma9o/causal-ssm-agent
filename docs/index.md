# Documentation Index

## Start Here

| Need | Open |
|---|---|
| Stage-by-stage pipeline walkthrough | [pipeline.md](pipeline.md) |
| Contributor and operator workflows | [guides/](guides/.md) |
| Cross-cutting runtime and modeling references | [reference/](reference/) |

## Artifact Owners

| Artifact or concept | Owning doc |
|---|---|
| `LatentModel` | [pipeline/01a-latent-model.md](pipeline/01a-latent-model.md) |
| `MeasurementModel`, `Indicator`, `CausalSpec`, `IdentifiabilityStatus` | [pipeline/01b-measurement-identifiability.md](pipeline/01b-measurement-identifiability.md) |
| `ObservationRecord`s and the encoded observation table (`data_for_model`) | [pipeline/02-indicator-extraction.md](pipeline/02-indicator-extraction.md) |
| `IndicatorAudit` and validation findings | [pipeline/03-extraction-validation.md](pipeline/03-extraction-validation.md) |
| `ModelSpec`, `LikelihoodSpec`, `ParameterSpec`, `PriorProposal` | [pipeline/04-model-specification-priors.md](pipeline/04-model-specification-priors.md) |
| `ParametricIdResult` and `InferenceStructureResult` | [pipeline/04b-parametric-identifiability.md](pipeline/04b-parametric-identifiability.md) |
| `FittedArtifact` and post-fit diagnostics | [pipeline/05b-inference-diagnostics.md](pipeline/05b-inference-diagnostics.md) |
| `TreatmentEffect` and intervention tools | [pipeline/06-intervention-analysis.md](pipeline/06-intervention-analysis.md) |

## Cross-Cutting References

| Topic | Open |
|---|---|
| Artifact lineage, temporal layers, assurance surfaces | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md) |
| Compilation from Stage 4 outputs to executable SSM runtime | [reference/compilation.md](reference/compilation.md) |
| Continuous-time estimation and discretization | [reference/estimation.md](reference/estimation.md) |
| Inference-method selection and structural routing | [reference/inference-routing.md](reference/inference-routing.md) |

## Guides

| Workflow | Open |
|---|---|
| Local setup | [guides/dev_setup.md](guides/dev_setup.md) |
| TypeScript codegen from Python contracts | [guides/codegen.md](guides/codegen.md) |
| Integration testing | [guides/agentic_integration_testing.md](guides/agentic_integration_testing.md) |
| Evaluations | [guides/running_evals.md](guides/running_evals.md) |

## Route by Question

| Question | Open first |
|---|---|
| What does this stage emit? | The relevant file under `pipeline/` |
| What does this artifact mean? | The stage doc that introduces it |
| What assumptions constrain that artifact? | The matching page under `reference/` |
| How does time work across extraction, fitting, and interventions? | [reference/pipeline-dimensions.md](reference/pipeline-dimensions.md) |
