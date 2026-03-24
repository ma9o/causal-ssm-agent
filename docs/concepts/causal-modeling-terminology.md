# Causal Modeling Terminology 

The term `structural` is historically shared by the SEM and SCM trasitions, but it points to different layers of a model.

- In SEM, the `structural model` is the part of the model that specifies directional relations among endogenous variables, in contrast to the `measurement model`.
- In SCM, `structural equations` are the assignment mechanisms `X_i = f_i(Pa_i, U_i)` for endogenous variables.

These are closely related ideas, but for clarity in this project we separate:

| Description | Domain Primitive | Introduction Stage |
|---|---|---|
| The latent-to-latent DAG proposed from theory | [`LatentModel`](../pipeline/01a-latent-model.md#latent-model) | [Stage 1a](../pipeline/01a-latent-model.md) |
| The construct-to-observed mapping | [`MeasurementModel`](../pipeline/01b-measurement-identifiability.md#measurement-model) | [Stage 1b](../pipeline/01b-measurement-identifiability.md) |
| The combined latent, measurement, and identifiability handoff | [`CausalSpec`](../pipeline/01b-measurement-identifiability.md#causalspec) | [Stage 1b](../pipeline/01b-measurement-identifiability.md) |
| The equations, likelihoods, priors, and parameterization used for fitting | [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) | [Stage 4](../pipeline/04-model-specification-priors.md) |
