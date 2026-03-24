# Counterfactual Inference

Post-estimation causal effect computation. For the estimation pipeline that produces posterior samples, see [estimation.md](estimation.md). For how Stage 6 uses these tools interactively, see [Stage 6](../pipeline/06-intervention-analysis.md).

## Do-Operator on Steady State

After estimation, causal effects are computed via the do-operator on the continuous-time steady state:

1. **Baseline steady state:** Given posterior draws of drift A and continuous intercept c, compute eta\* = -A^{-1}c (the CT steady state).
2. **Intervention:** Apply do(X = x) by replacing the treatment variable's row in A with an identity constraint and solving the modified linear system.
3. **Treatment effect:** Compare do(treat = baseline + 1) vs baseline for the outcome variable.

This is vmapped over posterior draws to produce posterior distributions of causal effects, ranked by effect size.

4. **Forward simulation (optional):** For time-varying interventions or transient dynamics, `forward_simulate_intervention()` propagates the discrete-time system forward under a specified intervention schedule, producing full trajectories rather than just steady-state comparisons.

## Interpretation Guidance

Effects are estimated as relationships between constructs as measured through their indicators. Measurement error in indicators is absorbed into residual variance. Interpret:

- **AR coefficients** as inertia in the construct
- **Cross-lag coefficients** as causal relationships between constructs
- **Time-invariant latents** as stable subject-level intercepts (see [A5](latent-model/assumptions.md#a5-time-invariant-latents-as-subject-level-static-states))

Causal interpretation requires that the DAG correctly captures the true causal structure and that all relevant confounders are included.
