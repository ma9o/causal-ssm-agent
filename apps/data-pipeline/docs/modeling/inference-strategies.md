# Inference Strategies for State-Space Models

This document covers causal-ssm-agent's likelihood backends and inference methods for continuous-time state-space models.

## The Marginalization Challenge

Given a state-space model with latent states **x**\_1:T and observations **y**\_1:T, parameter inference requires the marginal likelihood:

```
p(y_1:T | theta) = integral p(y_1:T, x_1:T | theta) dx_1:T
```

The latent states must be integrated out. For SSMs with T timesteps and n latent dimensions, this integral is over an (n x T)-dimensional space. The key to tractable inference is choosing the right marginalization strategy based on model structure.

### The Continuous-Time Formulation

causal-ssm-agent models continuous-time dynamics as a linear SDE:

```
d eta = (A eta + c) dt + G dW
```

where A is the drift matrix, c is the continuous intercept, and G is the diffusion Cholesky factor. For filtering and inference, this SDE is discretized to a discrete-time (DT) transition:

```
eta_t = Ad eta_{t-1} + cd + epsilon_t,    epsilon_t ~ N(0, Qd)
```

where Ad = exp(A dt), Qd is derived from the Lyapunov equation, and cd = A^{-1}(exp(A dt) - I)c. The discretization is computed by `causal_ssm_agent.models.ssm.discretization` and is shared across all likelihood backends.

## Likelihood Backends

Likelihood backends live in `src/causal_ssm_agent/models/likelihoods/` and compute log p(y | theta) given the SSM parameters. They are plugged into the NumPyro model via `numpyro.factor()`.

### Kalman Filter (`kalman.py`)

**Class:** `KalmanLikelihood`

Computes the exact marginal likelihood via the prediction error decomposition for linear-Gaussian SSMs. Uses cuthbert's non-associative moments filter (`gaussian.moments` with `associative=False`) for numerically stable gradients -- the associative variant uses QR decomposition that can produce NaN gradients for ill-conditioned matrices.

**Applicable when:**
- Linear dynamics (drift matrix A, no state-dependent nonlinearity)
- Gaussian process noise (`diffusion_dist == "gaussian"`)
- Gaussian observation noise (`manifest_dist == "gaussian"`)

**Complexity:** O(T n^3) -- one Cholesky per timestep, no sampling variance.

**Implementation:** Pre-discretizes all T time intervals into batched (Ad, Qd, cd) arrays, computes Cholesky factors, then feeds them to cuthbert's `build_filter` / `filter` pipeline. Returns `states.log_normalizing_constant[-1]`.

### Bootstrap Particle Filter (`particle.py`)

**Class:** `ParticleLikelihood`

Universal likelihood backend via cuthbert's bootstrap particle filter. Handles arbitrary observation distributions (Gaussian, Poisson, Student-t, Gamma) and process noise families (Gaussian, Student-t). With a fixed RNG key the PF likelihood is a deterministic function of theta, making it compatible with gradient-based inference via `numpyro.factor()`.

**Applicable when:** Any model. This is the fallback when Kalman assumptions fail.

**Complexity:** O(T n P) where P is the particle count (default 200).

**Key optimization:** Pre-discretizes CT->DT parameters and pre-computes Cholesky factors for all T timesteps outside the particle loop, avoiding redundant O(n^3) work per particle.

**Automatic RBPF upgrade:** When dynamics are Gaussian (`diffusion_dist == "gaussian"`), `ParticleLikelihood` automatically delegates to Rao-Blackwell callbacks instead of bootstrap callbacks. This is transparent to callers.

**Resampling:** Uses a pure-JAX systematic resampling implementation (`jnp.searchsorted`) instead of cuthbert's built-in `pure_callback + numba` version, which does not support JVP and therefore blocks `jax.grad` / NUTS.

### Rao-Blackwell Particle Filter (`rao_blackwell.py`)

**Factory:** `make_rb_callbacks`

When dynamics are linear-Gaussian but observations are non-Gaussian, the Kalman filter can analytically marginalize the latent state *inside each particle*. Particles carry Kalman sufficient statistics (`RBState`: mean, covariance, predicted mean, predicted covariance) instead of point samples, which gives strictly lower variance than the bootstrap PF.

**Architecture:**
- `init_sample` returns `RBState(m0, P0, m0, P0)` -- no sampling
- `propagate_sample` runs a deterministic Kalman predict step, then a linearized (EKF-style) update to keep covariances bounded
- `log_potential` evaluates `log integral p(y|x) N(x|m_pred, P_pred) dx` via quadrature

**Quadrature options:**
- **Unscented** (default): 2n+1 sigma points, exact for polynomials up to degree 3
- **Gauss-Hermite**: Tensor-product quadrature with configurable points per dimension

**Observation models:** Gaussian (exact), Poisson (log-link), Student-t (location-scale), Gamma (log-link). Non-Gaussian cases use the quadrature integration for weights and EKF-style linearized updates for state conditioning.

## Inference Methods

The `fit()` dispatcher in `src/causal_ssm_agent/models/ssm/inference.py` routes to eight methods:

| Method | Key | Type | Likelihood | Best For |
|--------|-----|------|------------|----------|
| SVI | `"svi"` | Variational | Any | Fast approximate posterior, tolerates PF noise |
| NUTS | `"nuts"` | MCMC (HMC) | Kalman preferred | Exact posterior, linear-Gaussian models |
| Hess-MC^2 | `"hessmc2"` | SMC | Any | Multimodal posteriors, gradient-rich proposals |
| PGAS | `"pgas"` | Gibbs + CSMC | Direct (no PF) | Non-Gaussian obs, trajectory-aware inference |
| Tempered SMC | `"tempered_smc"` | SMC + tempering | Any | Robust bridging from prior to posterior |
| Laplace-EM | `"laplace_em"` | IEKS + Laplace | Laplace approx | Fast mode-finding, linear-ish models |
| Structured VI | `"structured_vi"` | Variational | Any | Trajectory-aware variational posterior |
| DPF | `"dpf"` | Learned proposal PF | Learned | Amortized proposals for repeated inference |
| NUTS-DA | `"nuts_da"` | MCMC (data augmentation) | Direct (no filter) | Exact Gaussian dynamics, moderate T, "just run NUTS" |

### SVI (default)

**Module:** `inference.py` (`_fit_svi`)

Stochastic Variational Inference via ELBO optimization. Fits an `AutoMultivariateNormal` (or `AutoNormal`, `AutoDelta`) guide to approximate the posterior. The key property is that SGD is designed for noisy gradients, so SVI naturally tolerates the gradient noise from particle filter likelihoods.

**Parameters:**
- `guide_type`: `"mvn"` (default), `"normal"`, or `"delta"`
- `num_steps`: Optimization iterations (default 5000)
- `learning_rate`: ClippedAdam step size (default 0.01)
- `num_samples`: Posterior samples drawn from the fitted guide (default 1000)

**When to use:** Default choice. Fastest wall-clock time. Good for exploratory analysis, model checking, and as initialization for more expensive methods.

**Limitations:** Approximate posterior (Gaussian family), may underestimate posterior variance, does not capture multimodality.

### NUTS

**Module:** `inference.py` (`_fit_nuts`)

NumPyro's No-U-Turn Sampler (HMC variant). Uses `init_to_median` initialization and supports dense mass matrix adaptation.

**Parameters:**
- `num_warmup` / `num_samples`: MCMC budget (default 1000/1000)
- `num_chains`: Parallel chains (default 4)
- `target_accept_prob`: Default 0.85
- `max_tree_depth`: Default 8
- `dense_mass`: Use full mass matrix (default False)

**When to use:** When the Kalman likelihood applies (linear-Gaussian) and you want exact posterior samples. The smooth, deterministic Kalman log-likelihood gives clean gradients for HMC. Also works with PF likelihood but may struggle with resampling discontinuities.

**Limitations:** Requires differentiable log-likelihood. PF resampling creates gradient noise that can cause divergences. Single mode only.

### Hess-MC^2

**Module:** `ssm/hessmc2.py` (`fit_hessmc2`)

SMC sampler with gradient-based change-of-variables L-kernels (Murphy et al. 2025). Proposals are always accepted -- quality is controlled through importance weight correction, not MH accept/reject.

**Proposal types:**
- `"rw"`: Random walk (Eq 28). No gradients needed.
- `"mala"`: First-order Langevin / MALA proposals (Eq 30-33). Uses gradient of log-posterior.
- `"hessian"`: Second-order proposals (Eq 39-41). Uses full D x D Hessian of log-posterior as mass matrix, with automatic fallback to first-order when the negative Hessian is not PSD.

**Key design choices:**
- **No tempering:** Unlike standard SMC, targets the full posterior from iteration 1. Gradient/Hessian-informed proposals provide sufficient exploration.
- **Full Hessian:** Uses the complete D x D Hessian, not a diagonal approximation. For typical SSM dimensions (D=5-30), the O(D^3) cost is negligible compared to PF likelihood evaluation.
- **Optional warmup:** `warmup_iters` initial iterations use RW proposals with tempered reweighting to prevent particle collapse from diffuse priors.
- **Particle recycling:** All post-warmup particles and weights are stored and pooled for the final resampling step (Eq 26).

**Parameters:**
- `n_smc_particles`: Number of parameter particles (default 64)
- `n_iterations`: SMC iterations (default 20)
- `proposal`: `"rw"`, `"mala"`, or `"hessian"` (default)
- `step_size`: Proposal epsilon (default 0.1)
- `adapt_step_size`: ESS-based adaptation (default True)
- `warmup_iters`: RW warmup before main proposal (default 0)

**When to use:** Multimodal posteriors, models where NUTS struggles with PF gradient noise. The Hessian proposals provide local curvature information that accelerates convergence.

### PGAS

**Module:** `ssm/pgas.py` (`fit_pgas`)

Particle Gibbs with Ancestor Sampling (Lindsten, Jordan & Schoen, 2014). Gibbs-alternates between two conditionals:

1. **Trajectory step:** Sample x\_{1:T} | theta, y via Conditional SMC (CSMC) with the PGAS kernel. One particle is pinned to the reference trajectory; ancestor sampling connects it to the particle history for path diversity.
2. **Parameter step:** Update theta | x\_{1:T}, y via block HMC/MALA. Given a fixed trajectory, the log-posterior decomposes into prior + initial state density + transition densities + observation densities -- all cheap to evaluate without running a particle filter.

**Enhancements over basic PGAS:**
- **Gradient-informed CSMC proposals:** Free particles use Langevin-shifted proposals that incorporate the observation gradient, pushing particles toward high-likelihood regions.
- **Locally optimal proposal:** For Gaussian observations, analytically computes p(x\_t | x\_{t-1}, y\_t) which incorporates observation information directly into the proposal (activated automatically).
- **Preconditioned block HMC:** Parameters are split into blocks by site name, each with independent step sizes and mass matrices adapted from the running chain. Uses the shared `hmc_step` from `mcmc_utils.py`.
- **Running mass matrix:** Weighted covariance from the theta chain, updated periodically during warmup.

**Parameters:**
- `n_outer`: Gibbs iterations (default 50)
- `n_csmc_particles`: Particles in CSMC (default 20)
- `n_mh_steps`: HMC/MALA steps per parameter update (default 5)
- `n_leapfrog`: Leapfrog steps (1 = MALA, >1 = HMC)
- `langevin_step_size`: Gradient shift in CSMC (default 0.0)
- `param_step_size`: HMC epsilon (default 0.1)
- `block_sampling`: Update blocks independently (default True)

**When to use:** Non-Gaussian observation models (Poisson, Student-t, Gamma) where the RBPF or bootstrap PF is needed for state filtering. The Gibbs structure avoids differentiating through the particle filter for parameter updates, sidestepping the gradient noise problem entirely.

### Tempered SMC

**Module:** `ssm/tempered_smc.py` (`fit_tempered_smc`)

Adaptive tempering with preconditioned HMC/MALA mutations. Bridges the prior-posterior gap via a tempering ladder beta\_0=0 -> beta\_K=1, where the target at level k is p(theta) p(y|theta)^{beta\_k}.

**Key features:**
- **Adaptive tempering:** ESS-based bisection (Dau & Chopin 2022) selects beta increments to maintain a target ESS ratio. Falls back to linear schedule when `adaptive_tempering=False`.
- **Waste-free recycling:** Resamples M = N / n\_mh\_steps particles, runs n\_mh\_steps mutations on each, keeps all intermediates to reconstruct N particles with no wasted computation.
- **Preconditioned HMC:** Mass matrix set to the weighted particle precision (inverse covariance), updated only when ESS is healthy (> N/4).
- **Pilot adaptation:** Tunes step size at beta=0 (prior) before tempering starts, using aggressive Robbins-Monro updates.
- **Multi-step leapfrog:** `n_leapfrog > 1` runs L-step leapfrog instead of MALA.

**Parameters:**
- `n_csmc_particles`: Number of parameter particles (default 20)
- `n_outer`: Max tempering levels (default 100)
- `n_mh_steps`: HMC mutations per level (default 10)
- `param_step_size`: Initial leapfrog epsilon (default 0.1)
- `adaptive_tempering`: Use ESS bisection (default True)
- `target_ess_ratio`: ESS target as fraction of N (default 0.5)
- `waste_free`: Use waste-free recycling (default True, requires N % n\_mh\_steps == 0)
- `n_leapfrog`: Leapfrog steps (1 = MALA, >1 = HMC)
- `target_accept`: MH acceptance target (default 0.44 for MALA, 0.65 for HMC)

**When to use:** When the prior-posterior gap is large (vague priors, complex likelihoods), or when other methods get stuck in local modes. The tempering schedule provides a smooth path from prior to posterior.

### Laplace-EM

**Module:** `ssm/laplace_em.py` (`fit_laplace_em`)

Iterated Extended Kalman Smoother (IEKS) finds the MAP latent trajectory, then a Laplace approximation provides the marginal likelihood. Can be used as a fast initialization for other methods or as a standalone approximate inference method via tempered SMC on the Laplace-approximated likelihood.

**When to use:** Fast mode-finding for approximately linear models. Good as a warm-start for structured VI or tempered SMC.

### Structured VI

**Module:** `ssm/structured_vi.py` (`fit_structured_vi`)

Variational inference with a backward-factored Gaussian family: q(z\_{1:T} | phi) = q(z\_T) prod q(z\_t | z\_{t+1}). This structured family captures temporal correlations that standard mean-field guides cannot. ELBO is optimized jointly over variational parameters phi and model parameters theta.

**When to use:** When SVI's mean-field assumption is too restrictive and you need trajectory-aware uncertainty. Can be initialized from Laplace-EM output.

### Differentiable Particle Filter (DPF)

**Module:** `ssm/dpf.py` (`fit_dpf`)

Learns a neural proposal network q\_phi(z\_t | z\_{t-1}, y\_t) by optimizing the VSMC bound on prior-predictive data. At inference time, the learned proposal replaces the bootstrap prior proposal, yielding lower-variance importance weights. Uses soft resampling during training for differentiability and standard systematic resampling at inference.

**When to use:** When the bootstrap proposal is a poor match for the filtering distribution (high-dimensional latent states, informative observations). Amortizes proposal learning across datasets.

### NUTS Data Augmentation (NUTS-DA)

**Module:** `ssm/nuts_da.py` (`fit_nuts_da`)

Data augmentation MCMC (Tanner & Wong 1987): augments the parameter space with all latent states eta\_{0:T} and samples the joint posterior p(theta, eta\_{0:T} | y\_{1:T}) using NUTS. This is the "everything is parameters" approach — standard in Stan/NumPyro. No particle filter or Kalman filter is used during sampling; NUTS explores the full joint space using gradient information from the Gaussian transition and observation densities.

**Parameterizations:**
- **Centered** (default): Samples eta\_t ~ N(A\_d @ eta\_{t-1} + c\_d, Q\_d) directly via `numpyro.contrib.control_flow.scan`. Works well with SVI warmstart even at high T.
- **Non-centered**: Samples standardized innovations eps\_t ~ N(0, I) and computes eta\_t deterministically. Better without warmstart when process noise dominates; worse when observations are informative.

**SVI warmstart** (enabled by default): Runs variational inference on the Kalman-likelihood model (~50D parameter space) to find good parameter estimates. A Kalman smoother then provides optimal state estimates. Combined, these initialize NUTS near the posterior mode, breaking the chicken-and-egg problem of mass matrix adaptation in high dimensions. Falls back to particle-filter SVI if Kalman SVI fails, then to pseudo-inverse state initialization.

**Parameters:**
- `num_warmup` / `num_samples`: MCMC budget (default 1000/1000)
- `num_chains`: Parallel chains (default 4)
- `target_accept_prob`: Default 0.80
- `max_tree_depth`: Default 10
- `centered`: Centered vs non-centered parameterization (default True)
- `svi_warmstart`: Enable Kalman SVI warmstart (default True)
- `svi_num_steps`: SVI optimization steps (default 2000)

**When to use:** Moderate T (up to ~500 timesteps), exact Gaussian dynamics, and you want the simplicity of "just run NUTS" without choosing a likelihood backend or tuning SMC particles. The SVI warmstart makes this practical for SSMs where naive NUTS would fail to initialize.

**Limitations:** Joint space dimension grows as O(n\_latent x T), so not suitable for very long time series. Requires Gaussian observation noise (non-Gaussian obs need PGAS or PF-based methods).

## Shared Infrastructure

### MCMC Utilities (`ssm/mcmc_utils.py`)

Shared by PGAS, tempered SMC, Laplace-EM, structured VI, and DPF:

- **`hmc_step`**: Generalized HMC/MALA with full mass matrix preconditioning via Cholesky factor. When `n_leapfrog=1`, reduces to preconditioned MALA. Includes MH accept/reject.
- **`compute_weighted_chol_mass`**: Computes Cholesky of the weighted precision matrix (inverse covariance) from a particle cloud, matching the Stan/NUTS convention.
- **`find_next_beta`**: ESS-based bisection for adaptive tempering schedules.

### Site Discovery and Matrix Assembly (`ssm/utils.py`)

Shared by all SMC-based methods (Hess-MC^2, PGAS, tempered SMC, Laplace-EM, structured VI, DPF):

- **`_discover_sites`**: Traces the NumPyro model once to discover sample sites (names, shapes, distributions, bijective transforms).
- **`_assemble_deterministics`**: Builds SSM matrices (drift, diffusion, lambda, etc.) from constrained parameter samples in pure JAX (no numpyro handlers), enabling vmapped evaluation over particle clouds.
- **`_build_eval_fns`**: Returns JIT-compatible `log_lik_fn(z)` and `log_prior_unc_fn(z)` that operate on flat unconstrained parameter vectors, with `jax.checkpoint` for memory-efficient gradient computation through long time series.

## Selection Guidance

**Start with SVI.** It is fast, tolerates any likelihood backend, and gives a reasonable posterior approximation for model checking.

**For publishable results with linear-Gaussian models:** Use NUTS with KalmanLikelihood. The exact likelihood gives clean gradients and NUTS provides gold-standard posterior samples with convergence diagnostics.

**For non-Gaussian observations (Poisson, Student-t, Gamma):**
- **PGAS** is the recommended method. The Gibbs structure separates trajectory sampling (CSMC) from parameter updates (block HMC), avoiding the need to differentiate through the particle filter. The locally optimal proposal (for Gaussian obs) or gradient-informed proposals (for non-Gaussian obs) improve mixing.
- **SVI** with ParticleLikelihood works as a fast alternative but gives an approximate posterior.

**For multimodal posteriors or difficult geometry:**
- **Tempered SMC** provides a smooth tempering path from prior to posterior, handling multimodality through the particle population.
- **Hess-MC^2** with Hessian proposals adapts to local curvature, which can be more efficient than MALA/RW proposals when the posterior is anisotropic.

**For robust "just works" inference:** Tempered SMC with adaptive tempering and waste-free recycling is the most robust option. It requires minimal tuning (the tempering schedule adapts automatically) and avoids the gradient noise issues that affect NUTS with particle likelihoods.

| Scenario | Recommended | Likelihood | Rationale |
|----------|-------------|------------|-----------|
| Linear-Gaussian, fast exploration | SVI | Kalman | Fastest, good enough for iteration |
| Linear-Gaussian, publication quality | NUTS | Kalman | Exact posterior, convergence diagnostics |
| Linear-Gaussian, moderate T, simple setup | NUTS-DA | Direct | Joint param+state, no filter needed |
| Non-Gaussian obs, moderate dimension | PGAS | Direct | No PF in parameter step, block HMC |
| Multimodal posterior | Tempered SMC | PF | Tempering explores modes |
| Highly anisotropic posterior | Hess-MC^2 | PF | Hessian-adapted proposals |
| Unknown difficulty, want robustness | Tempered SMC | PF | Adaptive tempering, waste-free |

## References

- Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov Chain Monte Carlo Methods. JRSS-B.
- Dau, H.-D., & Chopin, N. (2022). Waste-Free Sequential Monte Carlo. JRSS-B.
- Lindsten, F., Jordan, M. I., & Schon, T. B. (2014). Particle Gibbs with Ancestor Sampling. JMLR.
- Murphy, J. et al. (2025). Hess-MC^2: Sequential Monte Carlo Squared using Hessian Information and Second Order Proposals.
- Sarkka, S. (2013). Bayesian Filtering and Smoothing. Cambridge University Press.
- Driver, C. C., & Voelkle, M. C. (2018). Hierarchical Bayesian Continuous Time Dynamic Modeling. Psychological Methods.
