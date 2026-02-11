# PGAS Scaling Literature Review: High-Dimensional Parameter Spaces

**Date:** 2026-02-10
**Focus:** Scaling Particle Gibbs with Ancestor Sampling (PGAS) to high-dimensional parameter spaces (D=50+)

## Executive Summary

PGAS faces fundamental scaling challenges in high-dimensional parameter spaces. Recent literature (2023-2025) emphasizes:

1. **Parameter updates should use trajectory-conditional likelihood** (exact given CSMC trajectory), not marginal PF likelihood
2. **Gradient-based proposals** (MALA, preconditioned Langevin) significantly outperform random-walk in D>20
3. **Acceptance rates degrade with dimension** unless gradient information or block updates are used
4. **Alternatives to standard PGAS** include Enhanced SMC², Particle-MALA, PARIS particle Gibbs, and Controlled SMC
5. **Waste-free SMC** improves particle efficiency but doesn't eliminate exponential scaling with state dimension

---

## 1. Parameter Update Strategy: Marginal vs Trajectory-Conditional Likelihood

### Standard PGAS Approach

In PGAS, parameters are updated using the **trajectory-conditional likelihood** p(θ | x₁:T[n], y₁:T), which conditions on the sampled trajectory from the Conditional SMC (CSMC) step.

**Key papers:**
- [Lindsten et al. (2014): Particle Gibbs with Ancestor Sampling](https://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf)
- [Andrieu, Doucet, Holenstein (2010): Particle Markov chain Monte Carlo methods](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf)

### Implementation

The natural approach is to update θ conditional on the current state path x₁:n:
- **Conjugate case:** Sample directly from p(θ | x₁:n, y₁:n)
- **Non-conjugate case:** Use Metropolis-within-Gibbs with proposals based on the trajectory

**Why not marginal likelihood?** The PGAS algorithm's theoretical validity relies on the Gibbs structure: alternating between:
1. Sampling trajectory x₁:T given θ (via CSMC with ancestor sampling)
2. Sampling θ given trajectory x₁:T

Using the marginal likelihood p(y₁:T | θ) = ∫ p(y₁:T | x₁:T, θ) p(x₁:T | θ) dx₁:T would break this structure and require a different algorithm (e.g., PMMH).

---

## 2. MCMC Kernels for High-Dimensional Parameter Updates

### 2.1 Random Walk Metropolis-Hastings (Standard Baseline)

**Scaling properties:**
- Optimal acceptance rate: ~0.234 in high dimensions
- Step size must shrink as O(d⁻¹/²) for dimension d
- Mixing time scales poorly with dimension

**Performance in SMC²:**
The standard random-walk proposal in SMC² "faces challenges, particularly with high-dimensional parameter spaces" ([Enhanced SMC², 2024](https://arxiv.org/abs/2407.17296)).

### 2.2 MALA (Metropolis-Adjusted Langevin Algorithm)

**Advantages:**
- Uses gradient information: proposal ~ N(θ + ε²/2 ∇log p(θ|data), ε²I)
- Better scaling: optimal acceptance ~0.574, step size O(d⁻¹/⁶)
- Significantly faster mixing in high dimensions

**Limitations:**
- Requires differentiable likelihood (particle filter gradients have high variance)
- Can get stuck in local modes without tempering
- "Initializing with MALA yields only marginal gains or even degraded performance due to the lack of Metropolis-Hastings correction" in some SMC contexts ([arXiv:2506.01320](https://arxiv.org/pdf/2506.01320))

**Papers:**
- [Adaptive Tuning of HMC Within Sequential Monte Carlo (2021)](https://projecteuclid.org/journals/bayesian-analysis/advance-publication/Adaptive-Tuning-of-Hamiltonian-Monte-Carlo-Within-Sequential-Monte-Carlo/10.1214/20-BA1222.pdf)

### 2.3 HMC and NUTS

**HMC (Hamiltonian Monte Carlo):**
- Uses gradient + momentum to explore parameter space efficiently
- Requires tuning: step size ε, number of leapfrog steps L
- Mass matrix M (preconditioning) crucial for correlated parameters

**NUTS (No-U-Turn Sampler):**
- Adaptive variant of HMC implemented in Stan, NumPyro
- Automatically tunes ε, M, and dynamically adapts L using no-U-turn criterion
- During warmup, tunes mass matrix based on sample covariance
- Uses dual averaging for step size adaptation

**Scaling:**
- Mass matrix estimation requires sufficient warmup samples
- "NUTS iteratively tunes the mass matrix M, the number of leapfrog steps L and the step size ε in order to achieve a high ESJD (expected squared jumping distance)"

**Challenges with particle filters:**
- Gradient estimation from particle filters is noisy
- May reject many trajectories if likelihood estimates are volatile
- Works best with smooth likelihoods (e.g., Kalman filter)

**Papers:**
- [NUTS paper (2014)](https://sites.stat.columbia.edu/gelman/research/published/nuts.pdf)
- [NumPyro MCMC documentation](https://num.pyro.ai/en/latest/mcmc.html)

### 2.4 Preconditioned Langevin (Recent Advances)

**Preconditioned Crank-Nicolson Langevin (pCNL):**
- "When combined with the Langevin algorithm, its semi-implicit Euler formulation allows for efficient exploration in high-dimensional space"
- "When augmented with MH correction, the acceptance rate is significantly improved compared to vanilla MALA"
- Used in Ψ-Sampler (2024-2025)

**Papers:**
- [Ψ-Sampler (2025)](https://arxiv.org/abs/2506.01320)

### 2.5 Recommendations by Dimension

| Dimension | Recommended Kernel | Notes |
|-----------|-------------------|-------|
| D < 5 | Random walk MH | Simple, effective |
| 5 ≤ D < 20 | MALA | Gradient helps, not too expensive |
| 20 ≤ D < 50 | Preconditioned MALA or HMC | Mass matrix crucial |
| D ≥ 50 | NUTS or block updates | Full covariance adaptation needed |

---

## 3. Mass Matrix Estimation and Block Updates

### 3.1 Mass Matrix (Covariance) Estimation

**Purpose:** Precondition the parameter space to account for correlations and different scales.

**Approaches:**
- **Empirical covariance:** Estimate from recent MCMC samples during warmup
- **Fisher information:** Use observed or expected Fisher information matrix
- **Adaptive schemes:** Continuously update during sampling (with diminishing adaptation)

**Challenge in state space models:**
"The likelihood function is often very flat in the common scale factor, so the full set of variances may be close to being perfectly correlated, which can cause problems with optimization routines" ([State Space Models: Parameter Estimation](https://estima.com/webhelp/topics/ssm-parameterestimation.html)).

### 3.2 Block Updates

**Motivation:** Parameters often fall into natural groups (e.g., dynamics vs observation, scale vs location).

**Block-Correlated Pseudo Marginal:**
- "A correlated pseudo-marginal (CPM) approach for Bayesian inference in state space models has been developed that is based on filtering the disturbances, rather than the states"
- "Induces a correlation of approximately 1-1/G between the logs of the estimated likelihood at the proposed and current values"
- "A novel block version of PMMH that works with multiple particle filters has been proposed"

**Papers:**
- [Block-Correlated Pseudo Marginal Sampler (2024)](https://www.tandfonline.com/doi/full/10.1080/07350015.2024.2308109)
- [Bayesian Inference using Block and Correlated Pseudo Marginal Methods (2016)](https://arxiv.org/abs/1612.07072)

### 3.3 Practical Strategy

For PGAS with D=50+ parameters:

1. **Group parameters into blocks:**
   - Dynamics parameters (state transition)
   - Observation parameters (measurement model)
   - Initial state parameters

2. **Within-block updates:**
   - Use gradient-based methods (MALA/HMC) with block-specific mass matrices
   - Estimate mass matrix from recent trajectory-conditional samples

3. **Cross-block updates:**
   - Occasionally update all parameters jointly to capture cross-block correlations
   - Use correlated proposals that share particle filter evaluations

---

## 4. Scaling Limitations of PGAS

### 4.1 Parameter Dimension Scaling

**Theoretical results:**
- Standard random-walk MCMC: acceptance rate decays exponentially with dimension without proper scaling
- MALA/HMC: better scaling but still require careful tuning

**Empirical observations:**
- PGAS works well for D < 10 parameters with standard random walk
- D = 10-20: gradient-based methods recommended
- D > 50: requires sophisticated proposals (preconditioned MALA, NUTS) or block updates

**Key limitation:**
"The effectiveness of the correlated PMMH (CPMMH) approach degrades as the observation variance and state dimension increases" ([Augmented pseudo-marginal MH, 2022](https://link.springer.com/article/10.1007/s11222-022-10083-5)).

### 4.2 Time Series Length Scaling

**Particle count scaling:**
"It is close to optimal to let the number of particles N scale at least linearly with T (the number of observations)" ([Andrieu, Doucet papers](https://www.stats.ox.ac.uk/~doucet/smc_resources.html)).

**Path degeneracy:**
- Standard CSMC suffers from path degeneracy: particles coalesce to single ancestor
- "Backward simulation approaches lead to a method which is much more robust to a low number of particles as well as a large number of observations"
- Ancestor sampling (PGAS) significantly mitigates this: "enables fast mixing even when using seemingly few particles"

**Practical recommendation:**
- For T=100 observations: N ≥ 50-100 particles
- For T=1000 observations: N ≥ 200-500 particles
- Depends on model complexity and state dimension

**Papers:**
- [Lindsten et al. on backward simulation (2011)](https://arxiv.org/abs/1110.2873)

### 4.3 State Dimension vs Parameter Dimension

**Critical distinction:**
- **State dimension** (e.g., 200+ latent variables): Particle filter suffers exponential curse of dimensionality
- **Parameter dimension** (e.g., 50+ static parameters): MCMC proposal must be carefully designed

**State dimension curse:**
"The number of particles required for a successful particle filter scales exponentially with the problem size. For a simple example with independent Gaussian states and observations, simulations indicate that the required ensemble size scales exponentially with state dimension" ([Obstacles to High-Dimensional Particle Filtering, 2008](https://www.stat.berkeley.edu/~bickel/Snyder%20et%20al%202008.pdf)).

Example: "The particle filter requires at least 10¹¹ members when applied to a 200-dimensional state" ([Particle filters for high-dimensional geoscience, 2019](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3551)).

**Mitigation for high state dimensions:**
- Localization schemes
- Ensemble Kalman filters (not particle-based)
- Auxiliary particle filters with better proposals

---

## 5. Alternatives to Standard PGAS

### 5.1 Enhanced SMC² with Gradient-Based Proposals

**Method:** Uses differentiable particle filters to get gradients of the likelihood w.r.t. parameters, then incorporates these into Langevin proposals within SMC².

**Key innovation:**
- "Harnesses first-order gradients derived from a CRN-PF (Common Random Numbers Particle Filter) using PyTorch's automatic differentiation"
- "Gradients leveraged within a Langevin proposal **without accept/reject**"
- "Can result in a higher effective sample size and more accurate parameter estimates compared to random-walk"

**Computational efficiency:**
- Parallelized with MPI
- O(log₂ N) time complexity
- "Achieves a 51x speed-up when compared to a single core using 64 computational cores"

**Papers:**
- [Enhanced SMC²: Leveraging Gradient Information from Differentiable Particle Filters (2024)](https://arxiv.org/abs/2407.17296)

### 5.2 Particle-MALA and Particle-mGRAD

**Particle-MALA:**
- "Spreads N particles locally around the current state using gradient information"
- Combines benefits of gradient-based exploration with particle diversity

**Particle-mGRAD:**
- "Additionally incorporates conditionally Gaussian prior dynamics into the proposal"
- "Interpolates between CSMC and Particle-MALA, resolving the 'tuning problem' of choosing between them"

**Application:** High-dimensional state-space models where standard CSMC degenerates.

**Papers:**
- [Particle-MALA and Particle-mGRAD (2024)](https://arxiv.org/abs/2401.14868)

### 5.3 PARIS Particle Gibbs (ICML 2023)

**Method:** Parisian Particle Gibbs (PPG) is a PaRIS (Particle Rapid Incremental Smoother) algorithm driven by conditional SMC moves.

**Key advantage:**
- "Bias-reduced estimates" of smoothing expectations
- Crucial for Maximum Likelihood Estimation (MLE) and Markov Score Climbing (MSC)
- "New bounds on bias and variance as well as deviation inequalities"
- "Implicit Rao-Blackwellization"

**Application:** Learning both states and parameters in non-linear state-space models.

**Papers:**
- [State and parameter learning with PARIS particle Gibbs (ICML 2023)](https://arxiv.org/abs/2301.00900)

### 5.4 Interacting Particle MCMC (iPMCMC)

**Problem with standard PGAS:**
"Path degeneracy can particularly adversely affect Particle Gibbs because conditioning on an existing trajectory means that the mixing of the Markov chain for early steps in the state sequence can become very slow when the particle set coalesces to a single ancestor."

**Solution:**
- Run **multiple PMCMC chains in parallel** with shared particle pool
- "Trade off exploration (SMC) and exploitation (CSMC) to achieve improved mixing"
- "Numerous candidate indices at each Gibbs update give significantly higher probability that an entirely new retained particle will be 'switched in'"

**Results:**
"Empirical results show significant improvements in mixing rates relative to both non-interacting PMCMC samplers and a single PMCMC sampler with an equivalent memory and computational budget."

**Papers:**
- [Interacting Particle Markov Chain Monte Carlo (2016)](https://arxiv.org/abs/1602.05128)

### 5.5 Controlled Sequential Monte Carlo (cSMC)

**Method:** Frames SMC proposal design as an optimal control problem.

**Key idea:**
- Cost functional: KL divergence from optimal proposals to target distributions
- Optimal proposals specified by solution to dynamic programming recursion
- "Iterative scheme building upon existing algorithms in econometrics, physics, and statistics"

**Algorithm:**
"The final algorithm alternates between performing a twisted SMC and approximate dynamic programming to find the next policy refinement."

**Application:** Works for both state space models (dynamic) and static models (parameter estimation).

**Papers:**
- [Controlled Sequential Monte Carlo (Heng, Bishop, Deligiannidis, Doucet, 2019)](https://arxiv.org/abs/1708.08396)
- [Probabilistic Inference in Language Models via Twisted SMC (2024)](https://arxiv.org/abs/2404.17546)

---

## 6. Waste-Free SMC: Particle Count vs Dimension

### 6.1 Standard SMC Inefficiency

**Problem:** Standard SMC samplers:
1. Propagate N particles forward
2. Run M MCMC steps per particle
3. Discard intermediate MCMC outputs
4. Only use final M-th iterate from each particle

This "wastes" N × (M-1) intermediate samples.

### 6.2 Waste-Free SMC Solution

**Method:**
- **Use ALL intermediate MCMC steps as particles**
- Total particles at each iteration: N × M
- More efficient use of computational budget

**Key insight:**
"Chaos propagation theory (Del Moral, 2004) says that when M ≥ N, M resampled particles behave essentially like M independent variables that follow the current target distribution."

**Papers:**
- [Waste-free Sequential Monte Carlo (Dau & Chopin, 2022)](https://arxiv.org/abs/2011.02328)
- [Waste-Free SMC in particles 0.3](https://statisfaction.wordpress.com/2021/10/25/particles-0-3-waste-free-smc-fortran-dependency-removed-binary-spaces/)

### 6.3 Particle Count Guidelines

**For waste-free SMC:**
- Total particles = N (resampled) × len_chain
- Example: N=20 particles with chain length 50 → 1000 effective particles per iteration

**Chain length selection:**
"Asymptotic variances depend on P (the chain length) in a non-trivial way. It is not clear how to choose P for optimal performance."

**Practical advice:**
- Start with moderate chain lengths (P=10-50)
- Monitor ESS (effective sample size)
- Increase if ESS/(N×P) is too low

### 6.4 Dimension Scaling (Still Exponential for State Space)

**Important:** Waste-free SMC does NOT solve the fundamental curse of dimensionality for particle filters in high-dimensional state spaces.

"Repeated resampling causes 'particle deprivation': most particles collapse onto a small region, leading to loss of diversity and poor posterior approximation **unless the number of particles scales exponentially with dimension**" ([High-Dimensional Particle Filter, 2019](https://arxiv.org/pdf/1901.10543)).

**For state dimension d:**
- Required particles: O(exp(cd)) for some constant c > 0
- Waste-free SMC helps with **parameter dimension** efficiency, not state dimension curse

---

## 7. Practical Recommendations

### 7.1 For D=50+ Parameters in PGAS

**Recommended workflow:**

1. **Use trajectory-conditional likelihood** in parameter update step
   - Sample θ ~ p(θ | x₁:T[n], y₁:T)
   - Exploit conjugacy where possible

2. **Choose parameter update kernel based on D:**
   - D < 20: MALA with empirical covariance preconditioning
   - 20 ≤ D < 50: Preconditioned MALA or HMC with block updates
   - D ≥ 50: NUTS with careful warmup, or block-diagonal mass matrix

3. **Block structure:**
   - Group parameters by type (dynamics, observation, scale)
   - Update blocks sequentially or with correlated proposals
   - Occasional joint updates to capture cross-block correlations

4. **Mass matrix estimation:**
   - Warm-up phase: 500-1000 iterations
   - Estimate block-wise covariances from trajectory-conditional samples
   - Use diagonal approximation if full covariance is unstable

5. **Particle count:**
   - Scale with time series length: N ≥ T/2 for T < 200
   - Monitor acceptance rates (target ~0.3-0.5 for MALA)
   - Monitor ESS of parameter samples

### 7.2 When to Use Alternatives

**Enhanced SMC² with gradients:**
- When you can implement differentiable particle filters (JAX, PyTorch)
- Large datasets (T > 1000)
- Need for parallel scaling

**Particle-MALA/mGRAD:**
- High state dimension (dx > 50)
- Gradient information available
- CSMC shows severe path degeneracy

**PARIS Particle Gibbs:**
- When estimating both states and parameters (joint learning)
- Need for bias control in smoothing expectations
- MLE or gradient-based parameter optimization

**Interacting Particle MCMC:**
- Have access to parallel compute resources
- Severe mixing problems with standard PGAS
- Can afford multiple independent particle filters

**Controlled SMC:**
- Complex proposal design needed
- Willing to invest in iterative tuning phase
- Both state space and static parameter problems

### 7.3 Diagnostic Checks

**For parameter MCMC:**
- Trace plots: check for mixing and stationarity
- Acceptance rate: 0.3-0.5 for MALA, 0.6-0.8 for NUTS
- ESS > 100 per chain after warmup
- R-hat < 1.1 across multiple chains

**For particle filter:**
- ESS at each time step: should be > N/2
- Particle weights: check for severe imbalance
- Path degeneracy: how far back do unique ancestors extend?

**For overall PGAS:**
- Joint log posterior trace: should mix well
- Parameter posterior correlations: check for pathologies
- Predictive checks: posterior predictive vs observed data

---

## 8. Key Papers by Author

### Andrieu, Doucet, Holenstein
- [Particle Markov chain Monte Carlo methods (2010)](https://www.stats.ox.ac.uk/~doucet/andrieu_doucet_holenstein_PMCMC.pdf) - **Foundational PMCMC paper**
- Established PMMH, Particle Gibbs, and theoretical foundations

### Lindsten, Schön, Jordan
- [Particle Gibbs with Ancestor Sampling (2014)](https://jmlr.org/papers/volume15/lindsten14a/lindsten14a.pdf) - **Foundational PGAS paper**
- [On the use of backward simulation (2011)](https://arxiv.org/abs/1110.2873)
- [Augmentation schemes for particle MCMC (2015)](https://link.springer.com/article/10.1007/s11222-015-9603-4)

### Chopin
- [SMC²: An efficient algorithm (2013)](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-9868.2012.01046.x)
- [Waste-free Sequential Monte Carlo (2022, with Dau)](https://arxiv.org/abs/2011.02328)
- [particles Python library](https://github.com/nchopin/particles) - **Practical implementation**

### Heng, Deligiannidis, Doucet
- [Controlled Sequential Monte Carlo (2019)](https://arxiv.org/abs/1708.08396)

### Recent Methods (2023-2025)
- [PARIS Particle Gibbs (ICML 2023, Cardoso et al.)](https://arxiv.org/abs/2301.00900)
- [Enhanced SMC² (2024, Rosato et al.)](https://arxiv.org/abs/2407.17296)
- [Particle-MALA and Particle-mGRAD (2024)](https://arxiv.org/abs/2401.14868)
- [Block-Correlated Pseudo Marginal (2024)](https://www.tandfonline.com/doi/full/10.1080/07350015.2024.2308109)

---

## 9. Open Research Questions

1. **Optimal mass matrix for trajectory-conditional posteriors:**
   - How to efficiently estimate covariance when conditioning on different trajectories each iteration?
   - Online vs batch estimation strategies?

2. **Gradient estimation from particle filters:**
   - CRN-PF vs score function estimators vs pathwise gradients
   - Bias-variance tradeoffs
   - When is gradient noise acceptable?

3. **Scaling laws:**
   - Precise characterization of acceptance rate vs dimension for PGAS with various kernels
   - Mixing time bounds as function of (D, T, N)

4. **Hybrid methods:**
   - Combining multiple PMCMC variants in single inference run
   - Adaptive switching between PGAS, PMMH, and gradient-based methods

5. **Tempering and annealing:**
   - Parallel tempering for multimodal parameter posteriors
   - Tempered transitions within PGAS framework

---

## References

All URLs are embedded as hyperlinks throughout the document above.

### Key Online Resources

- [SMC and Particle Methods Resources (Doucet)](https://www.stats.ox.ac.uk/~doucet/smc_resources.html)
- [particles Python library docs](https://particles-sequential-monte-carlo-in-python.readthedocs.io/)
- [NumPyro MCMC documentation](https://num.pyro.ai/en/latest/mcmc.html)
- [Stan HMC/NUTS documentation](https://mc-stan.org/docs/reference-manual/mcmc.html)

---

## Appendix: Search Queries Used

This literature review was compiled on 2026-02-10 using the following search queries:

1. "Particle Gibbs with Ancestor Sampling PGAS high dimensional parameters 2023 2024 2025"
2. "PGAS parameter update MCMC kernel MALA HMC NUTS scaling"
3. "Andrieu Doucet Lindsten Schon particle MCMC scaling limitations acceptance rate"
4. "SMC² particle MALA controlled SMC alternatives PGAS high dimensional"
5. "waste-free SMC particle count dimension scaling Chopin Del Moral"
6. "trajectory conditional likelihood vs marginal likelihood PGAS parameter update"
7. "PGAS mass matrix estimation block updates correlated parameters state space models"
8. "tempered SMC preconditioned MALA particle Gibbs gradient-based proposals 2024"
9. "PMMH particle marginal Metropolis Hastings vs PGAS comparison high dimensional"
10. "Chopin SMC2 state space parameter dimension scaling guidelines"
11. "controlled sequential Monte Carlo Heng twisted SMC state space inference"
12. "Lindsten Schon interacting particle MCMC correlated moves PGAS improvements"
13. "particle filter number of particles dimension scaling rule thumb 2024"
14. "PARIS particle Gibbs state parameter learning ICML 2023 Cardoso"
15. "differentiable particle filter gradient likelihood SMC2 enhanced Langevin 2024"

All searches performed via web search engines with access to academic preprint servers and journal databases.
