# Causal Modeling Terminology

The term `structural` is historically shared by the SEM and SCM traditions, but it points to different layers of a model.

- In SEM, the `structural model` is the part of the model that specifies directional relations among endogenous variables, in contrast to the `measurement model`.
- In SCM, `structural equations` are the assignment mechanisms `X_i = f_i(Pa_i, U_i)` for endogenous variables.

This project separates them:

| Description | Domain primitive | Owner stage |
|---|---|---|
| The latent-to-latent DAG proposed from theory | [`LatentModel`](latent-model/constructs-and-edges.md) | [Stage 1a](../pipeline/01a-latent-model.md) |
| The construct-to-observed mapping | [`MeasurementModel`](measurement-model/indicators.md) | [Stage 1b](../pipeline/01b-measurement-identifiability.md) |
| The combined latent, measurement, and identifiability handoff | [`CausalSpec`](causal-spec/identifiability.md) | [Stage 1b](../pipeline/01b-measurement-identifiability.md) |
| The equations, likelihoods, priors, and parameterization used for fitting | [`ModelSpec`](model-spec/parameters-likelihoods-and-priors.md) | [Stage 4](../pipeline/04-model-specification-priors.md) |
