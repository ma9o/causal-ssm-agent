# Prompting Best Practices for Model Specification

Guidelines for prompting LLMs to propose statistical models in the causal-ssm-agent pipeline.

## Avoid naming a modeling tradition

Do **not** frame prompts as "propose a DSEM" or "specify a SEM". Naming a tradition activates the LLM's priors about that tradition's defaults — for DSEM, this means linear Gaussian state-space models (Mplus-style Kalman filter), which collapses the observation model into Normal likelihoods regardless of actual data types.

Instead, prompt for **maximum fidelity to the data generating process**, constrained only by what is computationally feasible to estimate.

### Why this matters

Traditional DSEM (Mplus) is built on a linear Gaussian backbone:

- Counts get z-scored and treated as Normal, losing floor effects and overdispersion
- Binary/ordinal items go through probit thresholds with no native Poisson/NegBin support
- Link functions are implicit — everything is identity or probit, never log/logit as a deliberate choice
- Practitioners default to "standardize everything, assume Normal" because the tooling doesn't support alternatives

The measurement model becomes the weakest link: a sophisticated latent temporal structure sitting on distributional assumptions that don't match the DGP.

### What to do instead

Frame the task as **Bayesian generative modeling**:

- "Choose the distribution family that best matches each indicator's data type"
- "Select link functions appropriate for each distribution"
- Enumerate the valid options explicitly (Normal, Gamma, Bernoulli, Poisson, NegBin, Beta, OrderedLogistic, Categorical)
- Let the LLM reason about the DGP rather than pattern-match to a textbook tradition

This way the latent dynamics can still be linear (AR(1) + fixed effects) when that's appropriate, but the observation model respects actual data types — NegBin for overdispersed counts, Bernoulli for binary, OrderedLogistic for ordinal, Normal only for genuinely continuous measures.

### General principle

> Prompt for fidelity to the data generating process. Be constrained only by computability of estimation, never by the conventions of a named modeling tradition.
