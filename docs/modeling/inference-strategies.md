# Inference Strategies for State-Space Models

This document covers the inference methods available for continuous-time state-space models. For likelihood backend details and the CT-SDE formulation, see [estimation.md](estimation.md).

## The Marginalization Challenge

Given a state-space model with latent states **x**\_1:T and observations **y**\_1:T, parameter inference requires the marginal likelihood:

```
p(y_1:T | theta) = integral p(y_1:T, x_1:T | theta) dx_1:T
```

The latent states must be integrated out. For SSMs with T timesteps and n latent dimensions, this integral is over an (n x T)-dimensional space. The key to tractable inference is choosing the right marginalization strategy based on model structure.

## Inference Methods

The `fit()` dispatcher routes to nine methods:

### SVI (default)

Stochastic Variational Inference via ELBO optimization. Fits an auto-guide (multivariate normal, diagonal normal, or delta) to approximate the posterior. SGD naturally tolerates gradient noise from particle filter likelihoods.

**When to use:** Default choice. Fastest wall-clock time. Good for exploratory analysis, model checking, and as initialization for more expensive methods.

**Limitations:** Approximate posterior (Gaussian family), may underestimate posterior variance, does not capture multimodality.

### NUTS

NumPyro's No-U-Turn Sampler (HMC variant). Uses `init_to_median` initialization and supports dense mass matrix adaptation.

**When to use:** When the Kalman likelihood applies (linear-Gaussian) and you want exact posterior samples. The smooth, deterministic Kalman log-likelihood gives clean gradients for HMC. Also works with PF likelihood but may struggle with resampling discontinuities.

**Limitations:** Requires differentiable log-likelihood. PF resampling creates gradient noise that can cause divergences. Single mode only.

### Hess-MC^2

SMC sampler with gradient-based change-of-variables L-kernels (Murphy et al. 2025). Proposals are always accepted — quality is controlled through importance weight correction, not MH accept/reject. Supports random walk, MALA, and full Hessian proposals. The Hessian provides local curvature information that accelerates convergence in anisotropic posteriors.

**When to use:** Multimodal posteriors, models where NUTS struggles with PF gradient noise.

### PGAS

Particle Gibbs with Ancestor Sampling (Lindsten, Jordan & Schoen, 2014). Gibbs-alternates between:

1. **Trajectory step:** Sample x\_{1:T} | theta, y via Conditional SMC (CSMC) with the PGAS kernel.
2. **Parameter step:** Update theta | x\_{1:T}, y via block HMC/MALA. Given a fixed trajectory, the log-posterior decomposes into cheap densities without running a particle filter.

Enhancements: gradient-informed CSMC proposals, locally optimal proposal for Gaussian observations, preconditioned block HMC with running mass matrix.

**When to use:** Non-Gaussian observation models (Poisson, Student-t, Gamma) where the Gibbs structure avoids differentiating through the particle filter for parameter updates, sidestepping gradient noise entirely.

### Tempered SMC

Adaptive tempering with preconditioned HMC/MALA mutations. Bridges the prior-posterior gap via a tempering ladder beta\_0=0 -> beta\_K=1. Supports ESS-based adaptive tempering (Dau & Chopin 2022), waste-free recycling, and multi-step leapfrog.

**When to use:** When the prior-posterior gap is large (vague priors, complex likelihoods), or when other methods get stuck in local modes. The tempering schedule provides a smooth path from prior to posterior.

### Laplace-EM

Iterated Extended Kalman Smoother (IEKS) finds the MAP latent trajectory, then a Laplace approximation provides the marginal likelihood. Can be used as a fast initialization for other methods or standalone.

**When to use:** Fast mode-finding for approximately linear models. Good as a warm-start for structured VI or tempered SMC.

### Structured VI

Variational inference with a backward-factored Gaussian family: q(z\_{1:T} | phi) = q(z\_T) prod q(z\_t | z\_{t+1}). This structured family captures temporal correlations that standard mean-field guides cannot.

**When to use:** When SVI's mean-field assumption is too restrictive and you need trajectory-aware uncertainty.

### Differentiable Particle Filter (DPF)

Learns a neural proposal network q\_phi(z\_t | z\_{t-1}, y\_t) by optimizing the VSMC bound. At inference time, the learned proposal replaces the bootstrap prior proposal, yielding lower-variance importance weights.

**When to use:** When the bootstrap proposal is a poor match for the filtering distribution (high-dimensional latent states, informative observations).

### NUTS Data Augmentation (NUTS-DA)

Data augmentation MCMC: augments the parameter space with all latent states eta\_{0:T} and samples the joint posterior p(theta, eta\_{0:T} | y\_{1:T}) using NUTS. No particle filter or Kalman filter is used during sampling. Supports centered and non-centered parameterizations, with optional SVI warmstart via Kalman smoother.

**When to use:** Moderate T (up to ~500 timesteps), exact Gaussian dynamics, and you want the simplicity of "just run NUTS" without choosing a likelihood backend or tuning SMC particles.

**Limitations:** Joint space dimension grows as O(n\_latent x T). Requires Gaussian observation noise.

## Selection Guidance

**Start with SVI.** It is fast, tolerates any likelihood backend, and gives a reasonable posterior approximation for model checking.

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
