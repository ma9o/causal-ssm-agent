# Inference Routing for State-Space Models

The inference methods available for continuous-time state-space models, the design axes that distinguish them, and the structural routing logic that selects a method based on model properties. For likelihood backend details and the CT-SDE formulation, see [estimation.md](estimation.md).

Within the pipeline artifact lineage, this document explains how the fitted runtime chooses or exposes inference behavior after functional specification and compilation. For the cross-cutting pipeline map, see [pipeline-dimensions.md](pipeline-dimensions.md).

## The Marginalization Challenge

Given a state-space model with latent states **x**\_1:T and observations **y**\_1:T, parameter inference requires the marginal likelihood:

```text
p(y_1:T | theta) = integral p(y_1:T, x_1:T | theta) dx_1:T
```

The latent states must be integrated out. For SSMs with T timesteps and n latent dimensions, this integral is over an (n x T)-dimensional space. How to handle this integral -- and then how to explore the resulting parameter posterior -- are the two fundamental choices in SSM inference. These choices are not independent: the state marginalization method determines the gradient quality available for parameter exploration.

## Three Design Axes

Every inference method makes three design choices:

- **Axis A** — how to handle the latent-state integral (marginalize, augment, or Gibbs)
- **Axis B** — how to compute the marginal likelihood when marginalizing (Kalman, IEKS, PF, learned)
- **Axis C** — how to explore the parameter posterior (MCMC, VI, SMC)

These axes are conceptually distinct but not independent. Two dependencies structure the design space:

1. **A → C**: Augment and Gibbs both force C = MCMC. Augment creates an O(n\_latent × T)-dimensional sampling problem where VI and SMC are impractical (VI discards the exactness that motivates augmentation; SMC suffers weight degeneracy in high dimensions). Gibbs requires *sampling* from conditionals — replacing sampling with VI gives variational EM (a different algorithm class), and running SMC within each Gibbs sweep is better described as SMC². **Only Marginalize opens all three C options**, because it reduces the problem to inference on θ alone.
2. **B → C**: Within Marginalize, the likelihood computation method determines gradient quality, which constrains which parameter methods are viable. This is the binding constraint for the structural routing described below.

### Axis A: State-Parameter Coupling

How to deal with the nested integral over latent states.

| Strategy | Mechanism | Methods |
|----------|-----------|---------|
| **Marginalize** | Compute or approximate p(y\|theta) via a filter or analytical approximation, then do inference on theta alone | nuts, svi, tempered\_smc, hessmc2, laplace\_em, structured\_vi, dpf |
| **Augment** | Treat x\_{1:T} as parameters. Sample (theta, x\_{1:T}) jointly via NUTS on the augmented space. No filter needed. | nuts\_da |
| **Gibbs** | Alternate between p(x\|theta,y) via conditional SMC and p(theta\|x,y) via HMC. Each conditional avoids the hard integral. | pgas |

### Axis B: Marginal Likelihood Computation

When marginalizing (Axis A = Marginalize), how is p(y|theta) computed? The computation method determines gradient properties for parameter inference. This axis does not apply to Augment (no filter) or Gibbs (avoids the marginal likelihood entirely via conditional decomposition).

| Computation | Mechanism | Resulting gradients | Structural requirement | Methods |
|-------------|-----------|-------------------|----------------------|---------|
| **Closed-form** | Kalman filter | Exact, smooth | All emissions Gaussian + identity link + Gaussian diffusion | nuts, svi, tempered\_smc, hessmc2 |
| **Deterministic approx** | IEKS + Laplace | Smooth, approximate | Twice-differentiable emission log-density + linear dynamics | laplace\_em |
| **Stochastic estimate** | Bootstrap / RB particle filter | Noisy, stochastic | Universal (any model) | nuts, svi, tempered\_smc, hessmc2 |
| **Learned estimate** | Neural proposal PF or backward variational family | Lower variance, still stochastic (DPF) or variational bound (structured\_vi) | Universal (needs training phase) | dpf, structured\_vi |

Two critical observations:

1. **Our CT-LTI framework guarantees linear dynamics** (drift is always a matrix A). So "linear dynamics" is always satisfied. The IEKS path is blocked only when the emission log-density isn't twice-differentiable.

2. **All seven emission families have twice-differentiable log-densities** (Gaussian, Poisson, Student-t, Gamma, Bernoulli, Negative Binomial, Beta). So laplace\_em is structurally available for every non-Kalman model we support -- it just might not be *accurate* enough for highly non-Gaussian state posteriors (e.g., very sparse count data).

### Axis C: Parameter Posterior Method

Given the state handling from Axes A+B, how to explore the parameter posterior p(theta|y).

| Family | Exact (in limit)? | Tolerates noisy grad log p(y\|theta)? | Handles multimodality? | Methods |
|--------|-------------------|--------------------------------------|----------------------|---------|
| **MCMC** (NUTS/HMC) | Yes | No -- HMC/NUTS leapfrog needs smooth gradients. (Pseudo-marginal MH with PF is valid but slow; see Andrieu et al. 2010.) | No | nuts, nuts\_da, pgas |
| **VI** (SVI) | No (variational bound) | Yes -- SGD is designed for noise | No | svi |
| **SMC** (tempered, HessMC2) | Yes | Yes -- population-based | Yes | tempered\_smc, hessmc2, laplace\_em, structured\_vi, dpf |

### The B → C Constraint

Axis B determines gradient quality, which constrains which Axis C methods are viable:

| Likelihood (B) | Gradient quality | Viable parameter methods (C) |
|----------------|-----------------|------------------------------|
| **Closed-form** (Kalman) | Exact, smooth | All — MCMC is optimal (smooth target, exact gradients) |
| **Deterministic approx** (IEKS) | Smooth, approximate | MCMC and SMC both work. SMC preferred when multimodality is a concern |
| **Stochastic** (PF) | Noisy, discontinuous (resampling) | **MCMC inadvisable** — leapfrog divergences from resampling discontinuities. VI and SMC preferred |
| **Learned** (DPF, structured VI) | Lower variance, still stochastic | Same as stochastic — MCMC still inadvisable |

Combined with the A → C constraint (Augment and Gibbs force MCMC), this fully determines the viable methods. The structural routing reduces to: (1) default to Marginalize, (2) determine B from model structure, (3) select the best viable C given B. Augment and Gibbs are user overrides for specific needs, not structural routing targets.

## Method Taxonomy

The nine methods mapped to all three axes:

| Method | A: Coupling | B: Likelihood computation | C: Param method | Key advantage |
|--------|------------|--------------------------|----------------|---------------|
| `nuts` | Marginalize | Closed-form (Kalman) or stochastic (PF) | MCMC | Gold standard when gradients are smooth |
| `svi` | Marginalize | Closed-form (Kalman) or stochastic (PF) | VI | Fast, tolerates PF noise |
| `tempered_smc` | Marginalize | Closed-form (Kalman) or stochastic (PF) | SMC | Multimodal, robust to noise |
| `hessmc2` | Marginalize | Closed-form (Kalman) or stochastic (PF) | SMC (Hessian) | Curvature-adapted proposals |
| `laplace_em` | Marginalize | Deterministic approx (IEKS) | SMC | Avoids PF entirely for non-Gaussian |
| `structured_vi` | Marginalize | Learned (backward variational family) | SMC | Trajectory-aware state uncertainty |
| `dpf` | Marginalize | Learned (neural PF) | SMC | Lower-variance PF proposals |
| `nuts_da` | Augment | N/A (no filter) | MCMC | Simple "just run NUTS", no filter tuning |
| `pgas` | Gibbs | N/A (Gibbs conditional; CSMC samples states, not likelihood) | MCMC | Exact despite PF; gradient-free state updates |

## Structural Routing

The structural routing operates within A = Marginalize. (Augment and Gibbs force C = MCMC and are user overrides; see below.) The routing follows directly from the axis dependencies: determine B from model structure, then select C via the B → C constraint.

### Decision Tree

```text
SSMSpec + RBPartition
|
| A = Marginalize (structural default)
|
| Step 1: Determine B from model structure
|
+-- B = Closed-form (Kalman)?
|   partition.has_particle_block == False
|   All emissions Gaussian + identity link + Gaussian diffusion
|   |
|   | Step 2: Select C given B
|   | Exact, smooth gradients → all C options viable → pick MCMC
|   |
|   +-> "nuts"  [Closed-form, MCMC]
|       NUTS is the gold standard for smooth, differentiable targets.
|       Convergence diagnostics (R-hat, ESS, divergences) are
|       well-understood and trustworthy.
|
+-- B = Deterministic (IEKS/Laplace)?
    Non-Gaussian emissions (Poisson, Student-t, Gamma, Bernoulli, NegBin, Beta)
    |
    | Always available because:
    | - CT-LTI dynamics are always linear (IEKS requires this)
    | - All 7 emission families have C^2 log-densities (IEKS requires this)
    |
    | Step 2: Select C given B
    | Smooth, approximate gradients → MCMC and SMC both viable
    | → pick SMC (multimodality protection; non-Gaussian emissions
    |   frequently produce multimodal parameter posteriors)
    |
    +-> "laplace_em"  [Deterministic, SMC]
        IEKS finds the MAP state trajectory via Newton iterations.
        The Laplace approximation provides an analytical (noise-free)
        marginal likelihood without running a particle filter.
        Tempered SMC handles multimodality in the parameter posterior.
        O(T * D^3) per IEKS iteration, typically 3-8 iterations.
        No particle count to tune. No resampling noise.
```

### User Overrides

The structural routing picks the best default within A = Marginalize. Users can override to a different coupling strategy (A) or a different parameter method (C):

| Need | Override to | Axis change | Why |
|------|-----------|-------------|-----|
| Exact posterior from non-Gaussian model | `pgas` | A → Gibbs, C = MCMC | Gibbs structure avoids differentiating through PF |
| Simple setup, moderate T, Gaussian obs | `nuts_da` | A → Augment, C = MCMC | "Everything is parameters", no filter to configure |
| Fast exploration, any model | `svi` | C → VI | Fastest wall-clock, good for model iteration |
| Highly anisotropic posterior | `hessmc2` | C → SMC (Hessian) | Full Hessian proposals adapt to curvature |
| PF with severe particle degeneracy | `dpf` | B → Learned | Learned proposals reduce weight variance |
| Trajectory-aware state uncertainty | `structured_vi` | B → Learned | Backward-factored family captures temporal correlations |

## Method Reference

### NUTS

NumPyro's No-U-Turn Sampler (Hoffman & Gelman 2014). Uses `init_to_median` initialization, supports dense mass matrix adaptation.

**Axis position:** Marginalize + Closed-form (Kalman) + MCMC.

**When to use:** Kalman-eligible models (the structural default). The smooth, deterministic Kalman log-likelihood gives clean gradients. Also works with PF likelihood but may produce divergences from resampling discontinuities.

**Limitations:** Single mode. Requires differentiable log-likelihood. PF gradient noise causes divergences.

### SVI

Stochastic Variational Inference via ELBO optimization. Fits an auto-guide (multivariate normal, diagonal normal, or delta) to approximate the posterior. SGD naturally tolerates gradient noise from particle filter likelihoods.

**Axis position:** Marginalize + any likelihood computation + VI.

**When to use:** Fast exploration with any likelihood backend. Fallback when laplace\_em struggles with non-Gaussian emissions.

**Limitations:** Approximate posterior (Gaussian family). May underestimate posterior variance. Does not capture multimodality.

### Tempered SMC

Adaptive tempering with preconditioned HMC/MALA mutations (Dau & Chopin 2022). Bridges the prior-posterior gap via a tempering ladder beta\_0=0 --> beta\_K=1. Supports ESS-based adaptive tempering, waste-free recycling, and multi-step leapfrog.

**Axis position:** Marginalize + any likelihood computation + SMC.

**When to use:** When the prior-posterior gap is large, the posterior is multimodal, or other methods fail. The universal fallback.

### Hess-MC^2

SMC sampler with gradient-based change-of-variables L-kernels (Murphy et al. 2025). Proposals are always accepted; quality is controlled through importance weight correction, not MH accept/reject. Supports random walk, MALA, and full Hessian proposals. No tempering by design -- gradient- and Hessian-informed proposals provide sufficient exploration.

**Axis position:** Marginalize + any likelihood computation + SMC (Hessian).

**When to use:** Highly anisotropic posteriors where curvature information accelerates convergence.

### Laplace-EM

Iterated Extended Kalman Smoother (IEKS) finds the MAP latent trajectory, then a Laplace approximation provides the marginal likelihood. The outer loop uses tempered SMC for parameter inference.

**Axis position:** Marginalize + Deterministic approx (IEKS) + SMC.

**When to use:** Non-Gaussian emissions with linear dynamics (the structural default for non-Kalman models). Avoids particle filter noise entirely via analytical state marginalization. O(T D^3) per IEKS iteration, typically 3-8 iterations.

**Limitations:** Laplace approximation quality degrades for highly non-Gaussian state posteriors (sparse counts, boundary probabilities).

### Structured VI

Variational inference with a backward-factored Gaussian family: q(z\_{1:T} | phi) = q(z\_T) prod q(z\_t | z\_{t+1}). Captures temporal correlations that standard mean-field guides cannot. Can be initialized from Laplace-EM output.

**Axis position:** Marginalize + Learned estimate (backward variational family) + SMC. Unlike DPF which learns a PF proposal and uses the normalizing constant as the likelihood estimate, structured\_vi learns an approximation to the full state posterior and uses the ELBO as a surrogate for the marginal likelihood.

**When to use:** When SVI's mean-field assumption is too restrictive and you need trajectory-aware uncertainty.

### Differentiable Particle Filter (DPF)

Learns a neural proposal network q\_phi(z\_t | z\_{t-1}, y\_t) by optimizing the VSMC bound on prior-predictive data. At inference time, the learned proposal replaces the bootstrap prior proposal.

**Axis position:** Marginalize + Learned estimate (neural PF) + SMC.

**When to use:** When the bootstrap proposal is a poor match for the filtering distribution (high-dimensional latent states, informative observations causing particle degeneracy).

### NUTS Data Augmentation (NUTS-DA)

Data augmentation MCMC (Tanner & Wong 1987): augments the parameter space with all latent states eta\_{0:T} and samples the joint posterior p(theta, eta\_{0:T} | y\_{1:T}) using NUTS. Supports centered and non-centered parameterizations with optional SVI + Kalman smoother warmstart.

**Axis position:** Augment + N/A + MCMC.

**When to use:** Moderate T (up to ~500 timesteps), Gaussian observations, and you want the simplicity of "just run NUTS" without choosing a likelihood backend.

**Limitations:** Restricted to Gaussian observations (raises for non-Gaussian). Data augmentation MCMC is valid in principle for any emission family, but non-Gaussian emissions (Poisson near zero, Bernoulli near boundaries) create difficult posterior geometry in the O(n\_latent x T)-dimensional augmented space -- funnels, ridges, and sharp curvature that cause NUTS divergences. The restriction is a practical reliability choice.

### PGAS

Particle Gibbs with Ancestor Sampling (Lindsten, Jordan & Schon, 2014). Gibbs-alternates between trajectory sampling (CSMC with gradient-informed proposals) and parameter updates (block HMC/MALA with preconditioned mass matrix).

**Axis position:** Gibbs + N/A (CSMC samples states, not likelihood) + MCMC.

**When to use:** Non-Gaussian observation models where you want exact posterior samples. The Gibbs structure means the parameter conditional p(theta|x,y) is cheap to evaluate (no marginal likelihood needed), sidestepping PF gradient noise entirely.

**Limitations:** Mixing between Gibbs sweeps can be slow. Requires tuning particle count for CSMC.

<!-- ## Design Note: Why Structural Routing, Not a PSIS Cascade

An alternative design would be a linear cascade: try MAP, validate with PSIS k-hat, escalate to Laplace, validate, escalate to SVI, validate, escalate to NUTS. This was the original approach explored in `notebooks/inference_cascade.ipynb`.

This fails for three reasons:

1. **The methods are not linearly ordered.** They sit in a three-dimensional space (coupling x state quality x param method). Tempered SMC and PGAS are not "more expensive NUTS" -- they solve different problems (multimodality, joint state-parameter inference).

2. **PSIS validates an approximation to a fixed target.** It answers "is this proposal close enough to the posterior?" But the choice between SVI vs NUTS vs tempered SMC is not about approximation quality to the same target. It's about which target formulation is tractable given the model structure.

3. **The branching point is structural, not diagnostic.** Whether you can use Kalman vs PF vs IEKS is determined at model construction time by emission families and coupling structure. The `graph_analysis.py` module already computes this. Runtime diagnostics validate a chosen method; they don't select between fundamentally different strategies.

PSIS k-hat does appear in two places, neither of which is method routing:
- **Within Pathfinder** (future): validates the Gaussian approximation before deciding whether to warm-start NUTS.
- **Post-hoc LOO-CV** (`InferenceResult.get_loo_diagnostics`): validates the model, not the inference method. -->

## References

- Andrieu, C., Doucet, A., & Holenstein, R. (2010). Particle Markov Chain Monte Carlo Methods. JRSS-B.
- Dau, H.-D., & Chopin, N. (2022). Waste-Free Sequential Monte Carlo. JRSS-B.
- Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn Sampler. JMLR.
- Lindsten, F., Jordan, M. I., & Schon, T. B. (2014). Particle Gibbs with Ancestor Sampling. JMLR.
- Murphy, J. et al. (2025). Hess-MC^2: Sequential Monte Carlo Squared using Hessian Information and Second Order Proposals.
- Sarkka, S. (2013). Bayesian Filtering and Smoothing. Cambridge University Press.
- Tanner, M. A., & Wong, W. H. (1987). The Calculation of Posterior Distributions by Data Augmentation. JASA.
- Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder: Parallel Quasi-Newton Variational Inference. JMLR.
