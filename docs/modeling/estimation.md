# Estimation Pipeline

This document describes the end-to-end estimation pipeline: from continuous-time SDE specification through discretization, likelihood computation, and Bayesian inference. For inference strategy selection rationale, see [inference-strategies.md](inference-strategies.md).

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
- `F` is the observation noise family -- Gaussian by default, but also Poisson (log-link), Student-t, or Gamma (log-link).

These parameter types are codified in `src/dsem_agent/models/likelihoods/base.py:CTParams` and `src/dsem_agent/models/likelihoods/base.py:MeasurementParams`.

## 2. Discretization (CT to DT)

Observations arrive at discrete (possibly irregular) times. Before filtering, the continuous-time system must be discretized for each inter-observation interval `dt`.

### Core equations

Given drift `A`, diffusion covariance `Q_c = G G'`, and continuous intercept `c`:

| Discrete quantity | Formula | Implementation |
|---|---|---|
| Discrete drift | `A_d = exp(A * dt)` | `jax.scipy.linalg.expm(A * dt)` |
| Asymptotic covariance | `A * Q_inf + Q_inf * A' = -Q_c` (Lyapunov equation) | `solve_lyapunov()` via Kronecker vectorization |
| Discrete process noise | `Q_d = Q_inf - A_d * Q_inf * A_d'` | `compute_discrete_diffusion()` |
| Discrete intercept | `c_d = A^{-1} * (A_d - I) * c` | `compute_discrete_cint()` via `jla.solve` |

The Lyapunov equation is solved by vectorization: `(I kron A + A kron I) vec(X) = -vec(Q_c)`, then reshaping `vec(X)` back to a matrix. An iterative Smith-iteration solver (`solve_lyapunov_iterative`) is available for near-singular systems.

### Matrix fraction decomposition

For joint numerical stability, `matrix_fraction_decomposition()` computes all three discrete quantities from a single augmented matrix exponential:

```
        [A   Q_c  c]              [A_d   *    c_d]
exp(dt *[0  -A'   0])  =  approx  [0    *      0 ]
        [0   0    0]              [0    0      1 ]
```

The discrete process noise `Q_d` is extracted from the upper-right block product with `A_d'`.

### Batched discretization

For a time series with T observations and potentially irregular intervals, `discretize_system_batched()` uses `jax.vmap` over the `dt` dimension to produce:

- `Ad`: `(T, n, n)` -- discrete drift matrices per timestep
- `Qd`: `(T, n, n)` -- discrete process noise covariances per timestep
- `cd`: `(T, n)` -- discrete intercepts per timestep (or None)

This is a key optimization: the `O(n^3)` matrix exponential and Lyapunov solve are identical across particles and only need to be computed once per timestep, not once per particle.

All discretization logic lives in `src/dsem_agent/models/ssm/discretization.py`.

## 3. SSMSpec and SSMModel

### SSMSpec (`src/dsem_agent/models/ssm/model.py:SSMSpec`)

A dataclass specifying the model structure. Each matrix parameter can be:

- **Fixed**: a `jnp.ndarray` value used as-is (not estimated).
- **`"free"`**: estimated from data with full parameterization.
- **`"diag"`** (diffusion, manifest_var, t0_var only): estimated but constrained to diagonal.
- **`None`** (cint, manifest_means only): parameter is absent (zeros).

Key fields:

| Field | Shape | Description |
|---|---|---|
| `n_latent` | scalar | Number of latent processes |
| `n_manifest` | scalar | Number of observed indicators |
| `drift` | `(n_l, n_l)` or `"free"` | Continuous drift matrix A |
| `diffusion` | `(n_l, n_l)` or `"free"` / `"diag"` | Diffusion Cholesky G |
| `cint` | `(n_l,)` or `"free"` / `None` | Continuous intercept c |
| `lambda_mat` | `(n_m, n_l)` or `"free"` | Factor loadings Lambda |
| `manifest_means` | `(n_m,)` or `"free"` / `None` | Manifest intercepts mu |
| `manifest_var` | `(n_m, n_m)` or `"free"` / `"diag"` | Measurement error Cholesky |
| `t0_means` | `(n_l,)` or `"free"` | Initial state means |
| `t0_var` | `(n_l, n_l)` or `"free"` / `"diag"` | Initial state covariance Cholesky |
| `diffusion_dist` | `NoiseFamily` | Process noise family (default: Gaussian) |
| `manifest_dist` | `NoiseFamily` | Observation noise family (default: Gaussian) |
| `hierarchical` | `bool` | Enable multi-subject hierarchical structure |
| `n_subjects` | `int` | Number of subjects |
| `indvarying` | `list[str]` | Parameters that vary across individuals |

### SSMModel (`src/dsem_agent/models/ssm/model.py:SSMModel`)

Wraps an `SSMSpec` and implements the NumPyro model function. Constructor arguments:

- `spec`: the `SSMSpec`
- `priors`: an `SSMPriors` instance (or defaults)
- `n_particles`: particle count for bootstrap PF (default 200)
- `pf_seed`: fixed RNG seed so PF likelihood is deterministic for NUTS
- `likelihood`: `"particle"` (universal) or `"kalman"` (exact, linear Gaussian only)

The `model()` method is a standard NumPyro model function. It:

1. Samples all free parameters from their prior distributions via `_sample_drift()`, `_sample_diffusion()`, `_sample_cint()`, `_sample_lambda()`, `_sample_manifest_params()`, `_sample_t0_params()`.
2. Converts Cholesky factors to covariance matrices.
3. Samples noise family hyperparameters (e.g., `obs_df` for Student-t).
4. Instantiates the likelihood backend (`KalmanLikelihood` or `ParticleLikelihood`).
5. Packs parameters into `CTParams`, `MeasurementParams`, `InitialStateParams`.
6. Calls `backend.compute_log_likelihood()` and injects the result via `numpyro.factor("log_likelihood", ll)`.

For hierarchical models, step 6 uses `_hierarchical_likelihood()` which `vmap`s over subjects, extracting subject-specific parameters and computing per-subject observation masks and time intervals.

## 4. SSMPriors

`SSMPriors` (`src/dsem_agent/models/ssm/model.py:SSMPriors`) is a dataclass of dicts, each specifying distribution hyperparameters:

| Prior | Distribution | Default | Notes |
|---|---|---|---|
| `drift_diag` | `Normal(mu, sigma)` | `mu=-0.5, sigma=1.0` | Auto-effects, then `abs`-constrained negative |
| `drift_offdiag` | `Normal(mu, sigma)` | `mu=0.0, sigma=0.5` | Cross-effects |
| `diffusion_diag` | `HalfNormal(sigma)` | `sigma=1.0` | Process noise scale |
| `diffusion_offdiag` | `Normal(mu, sigma)` | `mu=0.0, sigma=0.5` | Off-diagonal diffusion (full Cholesky) |
| `cint` | `Normal(mu, sigma)` | `mu=0.0, sigma=1.0` | Continuous intercept |
| `lambda_free` | `Normal(mu, sigma)` | `mu=0.5, sigma=0.5` | Free factor loadings |
| `manifest_means` | `Normal(mu, sigma)` | `mu=0.0, sigma=2.0` | Manifest intercepts |
| `manifest_var_diag` | `HalfNormal(sigma)` | `sigma=1.0` | Measurement error scale |
| `t0_means` | `Normal(mu, sigma)` | `mu=0.0, sigma=2.0` | Initial state means |
| `t0_var_diag` | `HalfNormal(sigma)` | `sigma=2.0` | Initial state variance scale |
| `pop_sd` | `HalfNormal(sigma)` | `sigma=1.0` | Hierarchical random-effect SD |

The mapping from these dicts to NumPyro distributions happens inside the `SSMModel._sample_*` methods. For example, `drift_diag` becomes `dist.Normal(priors.drift_diag["mu"], priors.drift_diag["sigma"]).expand([n_latent])`.

Hierarchical parameters use a non-centered parameterization: `theta_i = theta_pop + sigma_pop * z_i` where `z_i ~ Normal(0, 1)`. This improves NUTS sampling geometry.

## 5. Likelihood Computation

Both backends implement the `LikelihoodBackend` protocol (`src/dsem_agent/models/likelihoods/base.py:LikelihoodBackend`) and share a common signature: `compute_log_likelihood(ct_params, measurement_params, initial_state, observations, time_intervals, ...)`.

### Kalman backend (`src/dsem_agent/models/likelihoods/kalman.py:KalmanLikelihood`)

For linear Gaussian models. Computes the exact marginal likelihood via the prediction error decomposition.

1. Pre-discretizes all timesteps via `discretize_system_batched()`.
2. Computes Cholesky factors of `Q_d` and the (possibly inflated) `R` for each timestep.
3. Builds `model_inputs` dict with temporal leading dimension `(T, ...)`.
4. Calls `cuthbert.gaussian.moments.build_filter()` with linear callbacks:
   - `get_dynamics_params` returns `f(x) = F @ x + c` with Cholesky `chol_Q`.
   - `get_observation_params` returns `g(x) = H @ x + d` with Cholesky `chol_R`.
5. Runs `cuthbert.filtering.filter()` to get `states.log_normalizing_constant[-1]`.

The non-associative moments filter (`associative=False`) is used because cuthbert's associative Kalman filter uses QR decomposition internally, which produces NaN gradients on ill-conditioned matrices.

Missing data is handled by inflating the measurement variance to `1e10` for unobserved channels (`preprocess_missing_data` in `base.py`), so the filter effectively ignores them.

### Particle filter backend (`src/dsem_agent/models/likelihoods/particle.py:ParticleLikelihood`)

Universal backend for arbitrary noise families and nonlinear dynamics.

1. Pre-discretizes all timesteps and pre-computes `chol(Q_d)` for each timestep.
2. Selects callback strategy:
   - **Gaussian dynamics** (`diffusion_dist == "gaussian"`): Uses Rao-Blackwellized callbacks (`src/dsem_agent/models/likelihoods/rao_blackwell.py:make_rb_callbacks`). Each particle carries Kalman sufficient statistics (`RBState`) instead of point samples. The Kalman predict step is deterministic (no noise sampling), and observation weights are computed via sigma-point quadrature for non-Gaussian observations or analytically for Gaussian observations. This gives strictly lower variance than bootstrap PF.
   - **Non-Gaussian dynamics** (e.g., Student-t process noise): Uses bootstrap PF callbacks via `SSMAdapter`. Particles are point samples propagated through `mean + chol_Qd @ noise` with the appropriate noise family.
3. Packs pre-discretized arrays into `model_inputs` with leading temporal dimension `T`.
4. Calls `cuthbert.smc.particle_filter.build_filter()` with the selected callbacks and a pure-JAX systematic resampling function (`_systematic_resampling`).
5. Runs `cuthbert.filtering.filter()` with a fixed RNG key and returns `states.log_normalizing_constant[-1]`.

The fixed RNG key makes the PF likelihood a deterministic function of parameters, which is required for NUTS (gradient-based HMC). Resampling uses `jnp.searchsorted` on cumulative weights -- integer indices have zero gradient, so gradients flow through particle weights and propagation only.

### How `numpyro.factor()` integrates the likelihood

Neither backend defines an explicit observation distribution. Instead, `SSMModel.model()` calls:

```python
numpyro.factor("log_likelihood", ll)
```

This adds the log-likelihood scalar directly to the model's log-joint density. NumPyro's inference engines (NUTS, SVI) then use this log-joint for gradient computation. The `factor` site acts like an observed sample site with a custom log-probability.

## 6. SSMModelBuilder

`SSMModelBuilder` (`src/dsem_agent/models/ssm_builder.py:SSMModelBuilder`) bridges the high-level DSEM pipeline (`ModelSpec` / `PriorProposal`) to the low-level SSM estimation stack.

### Conversion heuristic: `_convert_spec_to_ssm()`

Translates a `ModelSpec` (from the orchestrator) into an `SSMSpec`:

1. **Manifest columns**: extracted from `model_spec.likelihoods[*].variable`.
2. **Latent dimension**: inferred from the count of `ParameterRole.AR_COEFFICIENT` parameters (minimum 1).
3. **Hierarchical structure**: enabled if `model_spec.random_effects` is non-empty; `n_subjects` counted from `data["subject_id"]`.
4. **Noise family**: any non-Gaussian likelihood distribution triggers the particle backend. The mapping from `DistributionFamily` to `NoiseFamily` is in `_DIST_TO_NOISE`.
5. **Lambda matrix**: defaults to identity `eye(n_manifest, n_latent)`.
6. **Defaults**: drift free, diffusion diagonal, CINT free, manifest_var diagonal, individual-varying `t0_means`.

### Prior conversion: `_convert_priors_to_ssm()`

Maps named `PriorProposal` dicts to `SSMPriors` fields by keyword heuristics:

- Parameters containing `"rho"` or `"ar"` -> `drift_diag` prior
- Parameters containing `"beta"` -> `drift_offdiag` prior
- Parameters containing `"sigma"` or `"sd"` -> `diffusion_diag` prior

### Workflow

```python
builder = SSMModelBuilder(model_spec=spec, priors=priors)
builder.build_model(X)      # creates SSMModel from spec + data
result = builder.fit(X)     # runs inference, returns InferenceResult
samples = builder.get_samples()
builder.summary()
```

The `fit()` method delegates to `ssm.inference.fit()`, which dispatches to the chosen backend (SVI by default).

## 7. Library Stack

The estimation pipeline composes four main libraries:

### JAX

Foundation layer. Provides:

- `jax.numpy` for array operations on CPU/GPU.
- `jax.scipy.linalg.expm` for matrix exponentials in discretization.
- `jax.vmap` for batching discretization over timesteps and subjects.
- `jax.grad` / automatic differentiation for NUTS and SVI gradient computation.
- `jax.lax.scan` for sequential filtering operations.
- `jax.checkpoint` for memory-efficient backpropagation through long time series (recomputes forward pass during backward instead of storing intermediates).

### NumPyro

Probabilistic programming layer built on JAX. Provides:

- `numpyro.sample()` for declaring latent random variables with priors.
- `numpyro.factor()` for injecting custom log-likelihood terms.
- `numpyro.deterministic()` for recording derived quantities.
- `numpyro.infer.NUTS` for Hamiltonian Monte Carlo.
- `numpyro.infer.SVI` with `AutoNormal` / `AutoMultivariateNormal` guides for variational inference.
- `numpyro.handlers` for model tracing, parameter substitution, and site blocking.

### cuthbert

Differentiable filtering library. Provides two backends:

- **`cuthbert.gaussian.moments`**: Non-associative Kalman filter. Takes callbacks for dynamics and observation moments, runs predict-update recursion, returns `log_normalizing_constant`. Used by `KalmanLikelihood`.
- **`cuthbert.smc.particle_filter`**: Bootstrap / Rao-Blackwell particle filter. Takes `init_sample`, `propagate_sample`, `log_potential` callbacks. Supports arbitrary pytree particle states (enabling `RBState` for Rao-Blackwellization). Returns `log_normalizing_constant` for likelihood estimation. Used by `ParticleLikelihood`.

Both are invoked through `cuthbert.filtering.filter()`.

### dynamax (indirect)

Not directly imported, but cuthbert's Kalman filter implementation follows the same conventions as the dynamax library for linear Gaussian SSMs (prediction-error decomposition, square-root form via Cholesky factors).

### Data flow summary

```
ModelSpec (orchestrator)
    |
    v
SSMModelBuilder._convert_spec_to_ssm()
    |
    v
SSMSpec + SSMPriors
    |
    v
SSMModel.model()                     [NumPyro model function]
    |
    +-- _sample_*()                   [NumPyro priors -> parameters]
    |
    +-- discretize_system_batched()   [JAX: CT -> DT for all timesteps]
    |
    +-- KalmanLikelihood              [cuthbert gaussian.moments]
    |   or ParticleLikelihood         [cuthbert smc.particle_filter]
    |
    +-- numpyro.factor("log_likelihood", ll)
    |
    v
inference.fit()
    |
    +-- SVI (default): ELBO optimization via ClippedAdam
    +-- NUTS: HMC with differentiable log-joint
    +-- Hess-MC2 / PGAS / Tempered SMC: advanced SMC backends
    |
    v
InferenceResult (posterior samples + diagnostics)
```

## References

- `src/dsem_agent/models/ssm/model.py` -- SSMSpec, SSMPriors, SSMModel
- `src/dsem_agent/models/ssm/discretization.py` -- CT-to-DT conversion
- `src/dsem_agent/models/ssm/inference.py` -- fit(), SVI, NUTS backends
- `src/dsem_agent/models/ssm/utils.py` -- shared SMC utilities
- `src/dsem_agent/models/likelihoods/base.py` -- CTParams, MeasurementParams, LikelihoodBackend protocol
- `src/dsem_agent/models/likelihoods/kalman.py` -- KalmanLikelihood
- `src/dsem_agent/models/likelihoods/particle.py` -- ParticleLikelihood, SSMAdapter
- `src/dsem_agent/models/likelihoods/rao_blackwell.py` -- Rao-Blackwell callbacks, RBState
- `src/dsem_agent/models/ssm_builder.py` -- SSMModelBuilder, model_spec_to_ssm_spec
