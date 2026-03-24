# Causal Modeling Terminology

The term `structural` is historically shared by the SEM and SCM trasitions, but it points to different layers of a model.

- In SEM, the `structural model` is the part of the model that specifies directional relations among endogenous variables, in contrast to the `measurement model`.
- In SCM, `structural equations` are the assignment mechanisms `X_i = f_i(Pa_i, U_i)` for endogenous variables.

These are closely related ideas, but for clarity in this project we separate:

| Description | Model | Introduction Stage |
|---|---|---|
| The latent-to-latent DAG proposed from theory | `LatentModel` / latent model | [Stage 1a](../pipeline/01a-latent-model.md) |
| The construct-to-observed mapping | measurement model | [Stage 1b](../pipeline/01b-measurement-identifiability.md) |
| Causal DAG | topological structure | [Stage 1a](../pipeline/01a-latent-model.md) |
| The equations, likelihoods, priors, and parameterization used for fitting | functional specification | [Stage 4](../model-runtime/functional-specification.md) |

