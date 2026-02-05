# Inference Strategies for State-Space Models

This document covers the theoretical background for dsem-agent's automatic inference strategy selection.

## The Marginalization Problem

Given a state-space model with latent states **x**₁:T and observations **y**₁:T, we want to compute the marginal likelihood:

```
p(y₁:T | θ) = ∫ p(y₁:T, x₁:T | θ) dx₁:T
```

This integral is intractable for most models. The key insight is that certain model structures admit analytical or efficient approximate solutions.

## Model Structure and Tractability

### Linear-Gaussian Models

When both dynamics and observations are linear-Gaussian:

```
x_t = A x_{t-1} + q_t,    q_t ~ N(0, Q)
y_t = H x_t + r_t,        r_t ~ N(0, R)
```

The Kalman filter computes p(y₁:T | θ) exactly in O(T·n³) time via recursive prediction-update steps. This is the gold standard when it applies.

### Nonlinear Dynamics, Gaussian Noise

When dynamics are nonlinear but noise remains Gaussian:

```
x_t = f(x_{t-1}) + q_t,   q_t ~ N(0, Q)
y_t = H x_t + r_t,        r_t ~ N(0, R)
```

Two approximation strategies:

**Extended Kalman Filter (EKF):** Linearizes f(·) around the current estimate using Jacobians. With JAX autodiff, Jacobians are automatic. Accuracy degrades with strong nonlinearity.

**Unscented Kalman Filter (UKF):** Propagates deterministic "sigma points" through f(·), recovering mean and covariance without explicit Jacobians. Captures second-order effects that EKF misses. Generally preferred when f(·) is smooth.

Both are O(T·n³) and integrate with NumPyro via differentiable log-likelihood.

### Non-Gaussian or Strongly Nonlinear Models

When Kalman approximations fail:

```
x_t = f(x_{t-1}) + q_t,   q_t ~ arbitrary
y_t ~ p(y | g(x_t))       # e.g., Poisson, Student-t
```

**Particle Filter (Sequential Monte Carlo):** Represents p(x_t | y₁:t) with weighted samples. Handles arbitrary nonlinearity and non-Gaussianity. Complexity O(T·n·P) where P is particle count.

For parameter inference, particle MCMC methods (PMMH, Particle Gibbs) embed the particle filter within MCMC, using the particle estimate of p(y | θ) as the likelihood.

## The M-path Concept (Birch)

Birch PPL introduced "M-paths" for automatic strategy selection. An M-path is a chain of random variables connected by linear-Gaussian relationships:

```
x₀ ~ N(μ₀, Σ₀)
    ↓ [A·]           # linear transformation
x₁ ~ N(A·x₀, Q)
    ↓ [H·]           # linear transformation  
y₁ ~ N(H·x₁, R)
```

When the PPL detects an M-path, it knows Kalman operations apply. The entire chain can be marginalized analytically.

**Breaking the M-path:**

```
x₁ ~ N(f(x₀), Q)     # f is nonlinear → EKF/UKF or particle
y₁ ~ N(H·x₁, R)      # still linear measurement
```

Any nonlinear or non-Gaussian link breaks the M-path at that point.

### Implications for dsem-agent

Unlike Birch (runtime graph analysis), our ModelSpec explicitly declares structure. Detection is simpler—we inspect the spec statically:

| Component | Check | Kalman OK |
|-----------|-------|-----------|
| Dynamics | `drift` has no state-dependent terms | ✓ |
| Process noise | `diffusion_dist == "gaussian"` | ✓ |
| Measurement | `lambda_mat` has no state-dependent terms | ✓ |
| Observation noise | `manifest_dist == "gaussian"` | ✓ |

If all checks pass → Kalman. If only dynamics are nonlinear → UKF. Otherwise → Particle.

## Joint Structure vs Component-wise Analysis

A subtlety: the optimal inference strategy depends on the *joint* posterior structure, not just individual components.

Consider a model with:

```
[linear-Gaussian block] → [nonlinear link] → [linear-Gaussian block]
```

The optimal strategy isn't "particle filter everywhere" but rather: marginalize each linear-Gaussian block analytically, use particles only at the boundaries. This is **Rao-Blackwellization**.

**Example:** Linear dynamics, Poisson observations.

- Naïve particle filter: O(P) particles over full state
- Rao-Blackwellized: each particle carries Kalman sufficient statistics for p(x|y,θ), particles only for observation model → far fewer particles needed

### Current Scope

dsem-agent currently uses whole-model strategy selection (Kalman vs UKF vs Particle). Rao-Blackwellization is deferred because:

1. Typical CT-SEM models have 2-10 latent states—particle filtering scales fine
2. JAX/GPU acceleration pushes practical limits higher
3. Implementation complexity is significant

Revisit if users encounter scaling issues with state dimension >15 or series length >1000.

## Non-Gaussian Observation Models

Linear dynamics with non-Gaussian observations (Poisson counts, Student-t errors) are a common case that breaks Kalman but doesn't require full particle filtering.

### Options

**Laplace Approximation / Iterated EKF:** Treat non-Gaussian observation as locally Gaussian. For Poisson with log-link, linearize around current state estimate and iterate to convergence. Essentially what INLA does.

**Scale-Mixture Augmentation:** Some distributions decompose as Gaussian mixtures. Student-t is Gaussian with gamma-distributed precision. Augment state with auxiliary variables → conditionally Gaussian → Kalman applies.

**Particle Filter:** Always correct, just potentially slower. cuthbert handles arbitrary observation models.

### Current Scope

We default to particle filtering for non-Gaussian observations. The optimizations above are future work, triggered by performance needs.

## Library Mapping

| Strategy | Backend | Notes |
|----------|---------|-------|
| Kalman | dynamax | Exact, O(T·n³) |
| EKF | dynamax | JAX autodiff for Jacobians |
| UKF | dynamax | Sigma-point propagation, no Jacobians |
| Particle | cuthbert | Bootstrap filter, arbitrary models |
| PMMH | cuthbert + blackjax | Particle MCMC for parameters |

All backends are pure JAX, composable with NumPyro via `numpyro.factor`.

## References

- Murray & Schön (2018): [Delayed Sampling and Automatic Rao-Blackwellization](https://arxiv.org/abs/1708.07787)
- Särkkä (2013): Bayesian Filtering and Smoothing
- Driver & Voelkle (2018): Hierarchical Bayesian Continuous Time Dynamic Modeling
- [Birch automatic marginalization docs](https://birch-lang.org/concepts/automatic-marginalization/)
- [dynamax documentation](https://probml.github.io/dynamax/)
- [cuthbert repository](https://github.com/probml/cuthbert)
