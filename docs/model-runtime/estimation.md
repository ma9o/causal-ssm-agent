# Estimation Pipeline

This document describes the end-to-end estimation pipeline: from continuous-time SDE specification through discretization, likelihood computation, and Bayesian inference. For inference strategy selection rationale, see [inference-routing.md](inference-routing.md).

Within the pipeline artifact lineage, this document starts after Stage 4 has produced a [`ModelSpec`](../pipeline/04-model-specification-priors.md#modelspec) and the compilation path has produced an executable SSM runtime. For the cross-cutting pipeline map, see [../concepts/pipeline-dimensions.md](../concepts/pipeline-dimensions.md). If you need to locate an artifact owner quickly, see [../concepts/artifact-index.md](../concepts/artifact-index.md).

## 1. CT-SDE Formulation

The latent process is a multivariate Ornstein-Uhlenbeck SDE:

```
d eta(t) = (A * eta(t) + c) dt + G dW(t)
```

where:

- `A` is the `n_latent x n_latent` **drift matrix** controlling auto- and cross-regressive dynamics. Diagonal entries (auto-effects) are constrained negative for stability; off-diagonal entries (cross-effects) are unconstrained.
- `c` is the `n_latent x 1` **continuous intercept** (CINT), shifting the asymptotic mean away from zero.
- `G` is the `n_latent x n_latent` **diffusion Cholesky factor**, so `G G'` is the process noise covariance.
- `W(t)` is a standard Wiener process.

The observation (measurement) model is:

```
y(t) = Lambda * eta(t) + mu + epsilon,    epsilon ~ F(0, R)
```

where:

- `Lambda` is the `n_manifest x n_latent` **factor loading matrix** mapping latent states to observed indicators.
- `mu` is the `n_manifest x 1` **manifest intercept**.
- `R` is the `n_manifest x n_manifest` **measurement error covariance** (Cholesky-parameterized internally).
- `F` is the observation noise family -- Gaussian by default, but also Poisson (log-link), Student-t, Gamma (log-link), Bernoulli (logit-link), Negative Binomial (log-link), or Beta (logit-link).

## 2. Discretization (CT to DT)

Observations arrive at discrete (possibly irregular) times. Before filtering, the continuous-time system must be discretized for each inter-observation interval `dt`.

### Core equations

Given drift `A`, diffusion covariance `Q_c = G G'`, and continuous intercept `c`:

| Discrete quantity | Formula |
|---|---|
| Discrete drift | `A_d = exp(A * dt)` |
| Asymptotic covariance | `A * Q_inf + Q_inf * A' = -Q_c` (Lyapunov equation) |
| Discrete process noise | `Q_d = Q_inf - A_d * Q_inf * A_d'` |
| Discrete intercept | `c_d = A^{-1} * (A_d - I) * c` |

The Lyapunov equation is solved via Bartels-Stewart (Sylvester solver), which is O(n^3) vs O(n^6) for the Kronecker vectorization approach.

**Note on backward pass:** The forward pass uses Bartels-Stewart (Sylvester solver) at O(n^3). A custom VJP (`@jax.custom_vjp` on `solve_lyapunov`) uses implicit differentiation to compute gradients, but the adjoint Lyapunov equation is solved via Kronecker vectorization at O(n^6) because JAX lacks differentiation rules for Schur decomposition. For models with large latent dimension this backward pass can dominate gradient computation cost.

### Batched discretization

For a time series with T observations and potentially irregular intervals, the discretization is vmapped over the `dt` dimension to produce batched `(T, n, n)` discrete drift and noise matrices. The O(n^3) matrix exponential and Lyapunov solve are identical across particles and only need to be computed once per timestep, not once per particle.

## 3. Likelihood Computation

Both likelihood backends compute log p(y | theta) and inject it into the NumPyro model via `numpyro.factor()`, which adds the log-likelihood scalar directly to the model's log-joint density.

### Kalman backend

For linear-Gaussian models. Computes the exact marginal likelihood via the prediction error decomposition. Uses cuthbert's non-associative moments filter for numerically stable gradients.

**Applicable when:** Linear dynamics, Gaussian process noise, Gaussian observation noise.

**Complexity:** O(T n^3) -- one Cholesky per timestep, no sampling variance.

### Particle filter backend

Universal backend for arbitrary noise families and nonlinear dynamics. With a fixed RNG key the PF likelihood is a deterministic function of theta, making it compatible with gradient-based inference.

**Applicable when:** Any model. Fallback when Kalman assumptions fail.

**Complexity:** O(T n P) where P is the particle count.

**Automatic RBPF upgrade:** When dynamics are Gaussian but observations are non-Gaussian, the particle filter automatically delegates to Rao-Blackwell callbacks. Particles carry Kalman sufficient statistics instead of point samples, giving strictly lower variance than the bootstrap PF.

### Missing data handling

Missing observations are handled by inflating the measurement variance for unobserved channels, so the filter effectively ignores them.

## 4. Library Stack

The estimation pipeline composes three main libraries:

- **JAX**: Foundation layer. Array operations, matrix exponentials for discretization, vmap for batching, automatic differentiation for gradient-based inference, `lax.scan` for sequential filtering, `checkpoint` for memory-efficient backpropagation through long time series.

- **NumPyro**: Probabilistic programming layer. `sample()` for priors, `factor()` for custom log-likelihoods, `deterministic()` for derived quantities, NUTS for HMC, SVI with auto-guides for variational inference.

- **cuthbert**: Differentiable filtering library. Non-associative Kalman filter (`gaussian.moments`) and bootstrap/Rao-Blackwell particle filter (`smc.particle_filter`), both invoked through `cuthbert.filtering.filter()`.

### Data flow

```
[ModelSpec](../pipeline/04-model-specification-priors.md#modelspec) (orchestrator)
    |
    v
SSMModelBuilder.compile_inputs()
    |
    v
SSMSpec + SSMPriors
    |
    v
SSMModel.model()                     [NumPyro model function]
    |-- sample from priors
    |-- discretize CT -> DT
    |-- Kalman or Particle likelihood
    |-- numpyro.factor("log_likelihood", ll)
    |
    v
inference.fit()  -->  InferenceResult (posterior samples + diagnostics)
```

## 5. Counterfactual Inference (Do-Operator)

After estimation, causal effects are computed via the do-operator on the continuous-time steady state:

1. **Baseline steady state:** Given posterior draws of drift A and continuous intercept c, compute eta\* = -A^{-1}c (the CT steady state).
2. **Intervention:** Apply do(X = x) by replacing the treatment variable's row in A with an identity constraint and solving the modified linear system.
3. **Treatment effect:** Compare do(treat = baseline + 1) vs baseline for the outcome variable.

This is vmapped over posterior draws to produce posterior distributions of causal effects, ranked by effect size.

4. **Forward simulation (optional):** For time-varying interventions or transient dynamics, `forward_simulate_intervention()` propagates the discrete-time system forward under a specified intervention schedule, producing full trajectories rather than just steady-state comparisons.

## 6. Interpretation Guidance

Effects are estimated as relationships between constructs as measured through their indicators. Measurement error in indicators is absorbed into residual variance. Interpret:

- **AR coefficients** as inertia in the construct
- **Cross-lag coefficients** as causal relationships between constructs
- **Time-invariant latents** as stable subject-level intercepts (see [../concepts/assumptions.md](../concepts/assumptions.md) A5)

Causal interpretation requires that the DAG correctly captures the true causal structure and that all relevant confounders are included.

## References

- Driver, C. C., & Voelkle, M. C. (2018). Hierarchical Bayesian Continuous Time Dynamic Modeling. Psychological Methods.
- Sarkka, S. (2013). Bayesian Filtering and Smoothing. Cambridge University Press.
