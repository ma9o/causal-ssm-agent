# Inference Method Results

Empirical recovery benchmarks for each inference backend on a standardized 4-latent SSM.

## Test Problem

**Ground truth**: 4 latent processes (Stress -> Fatigue -> Focus -> Performance) with lower-triangular drift, diagonal diffusion, 6 Gaussian indicators (4 identity + 2 cross-loading), and free continuous intercepts.

```
Drift A (continuous-time):
           Stress  Fatigue  Focus   Perf
Stress     -0.80    0.00    0.00   0.00
Fatigue     0.30   -0.50    0.00   0.00
Focus      -0.20   -0.30   -0.60   0.00
Perf        0.00    0.00    0.40  -0.70

Diffusion SD:  [0.40, 0.30, 0.30, 0.25]
Cint:          [0.50, 0.00, 0.30, 0.00]
Obs noise SD:  [0.10, 0.10, 0.10, 0.10, 0.15, 0.15]
```

52 free parameters total (drift 4x4, diffusion 4, cint 4, lambda 8 free loadings, manifest means 6, manifest var 6, t0 means 4, t0 var 4). All observation noise parameters fixed at ground truth. Measurement model (lambda, manifest means, manifest var) estimated jointly.

Metrics:
- **RMSE**: root mean squared error of posterior means vs ground truth (over drift + diffusion + cint)
- **Corr**: Pearson correlation between posterior means and ground truth
- **Coverage**: fraction of parameters where 90% CI contains the true value

## PGAS (Particle Gibbs with Ancestor Sampling)

**Algorithm**: Gibbs-alternate between (1) CSMC sweep with ancestor sampling for latent trajectories and (2) MALA MH steps for parameters. Bootstrap proposals (no gradient shift). Systematic resampling via blackjax.

**Reference**: Lindsten, Jordan & Schon (2014), "Particle Gibbs with Ancestor Sampling."

### Results

| Setting | T | Iters | Warmup | CSMC particles | MH steps | Time | RMSE | Corr | Coverage |
|---------|---|-------|--------|----------------|----------|------|------|------|----------|
| Local CPU | 80 | 200 | 100 | 30 | 10 | 24s | 0.262 | 0.87 | 12/24 |
| A100 GPU | 200 | 500 | 250 | 50 | 15 | 521s | 0.258 | 0.89 | 14/24 |

### Parameter-level coverage (A100, T=200)

| Category | Coverage | Notes |
|----------|----------|-------|
| Drift diagonal | 2/4 | Fatigue and Perf biased negative (overestimated decay) |
| Drift off-diagonal | 9/12 | Best-recovered category |
| Diffusion SD | 1/4 | Hardest category; confounded with drift diagonal |
| Continuous intercept | 2/4 | Moderate recovery |

### Diagnostics

- MALA acceptance: ~37% mean (healthy range 25-50%)
- Step size adapted from 0.05 -> 0.021 via continuous adaptation
- Post-warmup adaptation prevents frozen chains (a failure mode when adaptation stops at warmup)

### Known limitations

1. **Drift-diffusion confounding**: In continuous-time SSMs, faster decay (larger |A_ii|) can be compensated by larger process noise (Q_ii). With bootstrap proposals and finite T, these parameters are partially non-identifiable. This explains the systematic negative bias in drift diagonal + poor diffusion coverage.

2. **Gradient proposal feedback loop**: Gradient-informed CSMC proposals (langevin_step_size > 0) create a positive feedback loop when observation noise is jointly estimated. The gradient nabla_x log g(y|x) = Lambda^T R^{-1}(y - Lambda*x - mu) grows inversely with R, pushing particles too close to observations -> MALA overestimates diffusion/decay -> R drops -> loop amplifies. Fix: default to bootstrap proposals; gradient proposals are opt-in with shift clipping.

3. **Scaling**: 521s on A100 for T=200, 500 iters. The bottleneck is the sequential CSMC sweep (T steps, not parallelizable). GPU helps with the vmap over particles but doesn't help with the time axis.

## Tempered SMC with Preconditioned MALA

**Algorithm**: Linear tempering (beta: 0->1) with preconditioned MALA mutations (1-step leapfrog HMC + MH correction). Pilot adaptation at beta=0, adaptive step size, weighted precision mass matrix. Marginalizes trajectories via PF (no CSMC conditioning).

### Development History

#### Run A: M=covariance (BUG)

Mass matrix set to covariance instead of precision. Three GPU runs with identical settings but different code states.

| Run | Coverage | RMSE | Corr | Final eps | Accept at beta=1 |
|-----|----------|------|------|-----------|-------------------|
| A1 | 13/24 | 0.507 | 0.772 | 0.0000 | 0.40 (deceptive) |
| A2 | 0/24 | 0.412 | 0.638 | 0.0000 | 0.00 |
| A3 | 11/24 | 0.361 | 0.646 | 0.0000 | 0.03 |

**Diagnosis:** With M=cov, the MALA position update is `z += eps * cov^{-1} * p`, giving noise variance `eps^2 * cov^{-1}` -- non-isotropic, tiny steps in wide directions, large in narrow. Step size adaptation fights this by driving eps to zero.

#### Run B: M=precision, target=0.50, rate=0.3

Fixed mass matrix to precision (inverse covariance). MALA noise becomes `eps * N(0, I)` in standardized space -- perfectly isotropic.

| Run | Coverage | RMSE | Corr | Final eps | Accept at beta=1 |
|-----|----------|------|------|-----------|-------------------|
| B1 | 21/24 | 0.160 | 0.929 | 0.0005 | 0.30 |

**Observation:** Dramatic improvement. eps still decays to 0.0005 near beta=1 because acceptance (0.44) consistently below target (0.50), so adaptation keeps shrinking.

#### Run C: M=precision, target=0.44, rate=0.1 (FINAL)

Lowered target acceptance to match natural equilibrium with PF noise. Slower adaptation rate prevents over-reacting to transient dips.

| Run | Coverage | RMSE | Corr | Final eps | Accept at beta=1 |
|-----|----------|------|------|-----------|-------------------|
| C1 | **22/24** | **0.183** | **0.882** | 0.020 | 0.28 |

**This is the committed configuration.**

Breakdown: drift diag **4/4**, drift off-diag 11/12, diffusion **4/4**, cint 3/4.

Misses: Perf->Stress off-diagonal (bias +0.42), Stress continuous intercept (bias -0.56). Both are weakly identified parameters with wide posteriors.

eps trajectory: 0.26 (pilot) -> 0.16 (beta=0.2) -> 0.10 (beta=0.5) -> 0.05 (beta=0.8) -> 0.02 (beta=1.0). Healthy decay reflecting genuinely harder posterior at higher beta, not adaptation pathology.

Settings: `N_SMC=128, K=200 (100 warmup), N_MH=15, N_PF=500, target_accept=0.44, adapt_rate=0.1, pilot=30 steps, B200 GPU, ~35 min`

### Key Lessons

1. **Mass matrix convention (CRITICAL)**: For MALA/HMC, M should be the **precision** (inverse covariance), not the covariance. This matches Stan/NUTS convention where "inverse mass matrix" = covariance, so M = precision.

2. **Step size adaptation**: Target acceptance ~0.44 works for MALA with noisy PF likelihood (lower than theoretical 0.574 due to gradient noise). Slow adaptation rate (0.1) prevents over-adapting to transient acceptance dips from PF noise.

3. **Tempering**: Linear schedule beta=k/n_outer is simple and robust. Guarded mass matrix: only update when ESS > N/4 to prevent degenerate covariance estimates.

## Hess-MC2 (SMC-squared with Hessian proposals)

Fundamentally fails at D=52. Importance-weighted proposals cannot bridge the prior-posterior gap in high dimensions -- ESS collapses to 1/N even with tempered warmup. Works at D=3 (1-latent LGSS) but does not scale.

## PMMH (Particle Marginal Metropolis-Hastings)

*Results pending.*

## Summary Table

| Method | Coverage | RMSE | Corr | Notes |
|--------|----------|------|------|-------|
| PGAS (Gibbs CSMC + MALA) | 14/24 | 0.258 | 0.89 | A100, drift-diffusion confounding |
| **Tempered SMC + MALA** | **22/24** | **0.183** | **0.88** | B200, precision preconditioning |
| Hess-MC2 (D=52) | fails | -- | -- | ESS collapse in high dimensions |
