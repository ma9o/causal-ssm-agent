# Parameter Identification: Location and Scale Anchors

This reference covers *parameter-level* identification in the compiled SSM: the rules that prevent groups of parameters from moving together along exact likelihood ridges. It complements [causal identifiability](../causal-design/identifiability.md), which asks whether a treatment effect is identifiable from the graph; this page asks whether the fitted model's parameters are identifiable from the data.

## The Anchor Invariant

Every retained construct has **exactly one location anchor** and **exactly one scale anchor**. Any additive shift or multiplicative rescaling of a latent state that the anchors do not pin must be absorbable by exactly one free parameter group — never by two, which would create an exact ridge, and never by zero, which would over-constrain the model into misfit.

Compilation enforces the invariant: [spec translation](../compilation.md) audits every construct and fails with aggregated errors when an anchor is missing. The guard test `tests/models/ssm/test_identification_anchors.py` enumerates the family and policy combinations.

## Location Anchors

A construct's location is pinned by the first applicable anchor:

- **Standardized channel.** Gaussian/Student-t identity-link indicators with additive-location support are auto-standardized (mean-centered, unit-scaled), and their observation intercept is fixed at 0. The centered data pin the latent level.
- **Dynamics.** A dynamic construct without equilibrium forcing relaxes toward a potential well pinned at 0, so its stationary level is 0 by construction.
- **Fixed t0 mean.** A time-invariant construct has no dynamics anchor; without a standardized channel its `t0_mean` is fixed at 0.

Channel-side location parameters are then identified *relative to* the anchor and stay free:

- observation intercepts (`manifest_mean_*`) on scalar families (Poisson, Bernoulli, Gamma, Beta, negative binomial) and on non-standardized location channels;
- ordered-logistic threshold bases (`obs_ordered_base`) — cutpoints are **not** centered, because centering both cancels the base out of the likelihood exactly and forces a second location anchor onto already-anchored constructs;
- categorical class intercepts (`obs_cat_intercepts`), with the baseline category pinned at zero logit.

Latent-side location parameters are gated on the standardized-channel anchor, mirroring ctsem's rule that the continuous intercept and the manifest means must not both be free[^driver2017]:

- `cint_*` (the free well center under `equilibrium_forcing`) activates only for constructs with a standardized channel;
- `t0_mean_*` for time-invariant constructs activates only under the same condition.

Without this gate, the free latent-side location and the channel-side location parameters move together on an exact additive ridge — and because structural edges act on raw states, the ridge propagates through every downstream construct whose center is free.

## Scale Anchors

- **Reference loading.** Each construct's reference indicator has its loading fixed to ±1 by polarity (the marker-variable convention[^bollen1989]). The fixed loading anchors scale only when the channel's link has a fixed scale: standardized identity channels pin scale in data units, ordinal and binary channels through the unit logistic scale, count channels through log-link curvature.
- **Categorical anchor slope.** Categorical channels never anchor scale through the loading: the free class slopes multiply the whole linear predictor, so a pinned loading is absorbed by them. For a construct measured *only* by categorical channels, the reference channel's first non-baseline slope is pinned to +1 instead (the nominal-response-model convention[^bock1972]), which anchors scale and sign in one move.

Consequences:

- loadings on categorical channels are always compiler-pinned (a free loading is exactly redundant with the slopes), and the loading prior surface deactivates when the LLM chooses a categorical likelihood;
- [reference indicators](../../pipeline/measurement-structure.md) are chosen by dtype tier — continuous, then ordinal, then binary/count, then categorical — so the strongest available anchor carries the construct's unit and the latent stays on the standardized scale the dynamics priors are authored under.

## Sign

Reflection symmetry (latent and its loadings jointly negated) is broken by the polarity-constrained loading signs; for all-categorical constructs, where polarity is meaningless, the pinned +1 anchor slope breaks it instead.

## Compile-Time Audit

Spec translation fails when any construct violates the invariant:

- a free equilibrium center or free static t0 mean without a standardized channel (no location anchor);
- no fixed loading on a non-categorical channel and no pinned categorical anchor slope (no scale anchor);
- a free loading on a categorical channel (exact redundancy with the slopes);
- a retained construct with no retained indicators (location and scale both unidentified — marginalize or drop it instead).

## Soft Ridges (Deliberately Not Gated)

The following pairs are weakly separated rather than exactly redundant; priors keep them proper and post-fit diagnostics (posterior contraction, power-scaling) are the right place to watch them:

- transient vs equilibrium location (`t0_mean` vs `cint`) when dynamics are slow relative to the observation span — avoided by the stationary initialization policy;
- static baseline-factor SDs (`tau`) in a single-subject fit, where one factor realization is observed and the variance is prior-dominated;
- observation dispersion vs latent volatility (negative-binomial `obs_r`, Gamma shape, Student-t `obs_df` against the diffusion SD), separated only through temporal autocorrelation;
- diffusion correlation (`cor`) vs reciprocal fast edges at coarse sampling;
- Hill `Emax` vs `EC50` when the input never spans the half-saturation point.

[^bollen1989]: Bollen, K. A. (1989). *Structural Equations with Latent Variables*. Wiley. [Bibliography entry](../bibliography.md)
[^bock1972]: Bock, R. D. (1972). Estimating Item Parameters and Latent Ability when Responses Are Scored in Two or More Nominal Categories. *Psychometrika*, 37(1), 29–51. [Bibliography entry](../bibliography.md)
[^driver2017]: Driver, C. C., Oud, J. H. L., & Voelkle, M. C. (2017). Continuous Time Structural Equation Modeling with R Package ctsem. *Journal of Statistical Software*, 77(5). [Bibliography entry](../bibliography.md)
