# Mplus Feature Parity Analysis

Comparison of Asparouhov, Hamaker & Muthén (2017) "Dynamic Structural Equation Models" (DSEM) against the causal-ssm-agent framework implementation.

**Reference:** Asparouhov, T., Hamaker, E. L., & Muthén, B. (2017). Dynamic Structural Equation Models. *Structural Equation Modeling: A Multidisciplinary Journal*. DOI: 10.1080/10705511.2017.1406803

---

## Core Architecture Comparison

### Paper's Three-Level Decomposition

```
Y_it = Y_{1,it} + Y_{2,i} + Y_{3,t}
       (within)  (person) (time)
```

- **Y_{1,it}**: Within-level deviation (person i at time t)
- **Y_{2,i}**: Person-specific component (time-invariant)
- **Y_{3,t}**: Time-specific component (person-invariant)

### Framework Implementation

**Two-level only:** Person-specific (`Y_{2,i}`) + within-level (`Y_{1,it}`). No time-specific random effects (`Y_{3,t}`).

This is appropriate for most ILD applications where time scales are not aligned across individuals.

---

## Intentionally Limited Features

These features from the paper are **excluded by design** in causal-ssm-agent.

### 1. AR(1) Only (No Higher-Order Lags)

| Paper | Framework |
|-------|-----------|
| Supports AR(L) for any L | AR(1) only (Assumption A3) |

**Rationale:** First-order Markov property enables 2-timestep unrolling for identification (per Jahn et al. 2025, arXiv:2504.20172). Higher-order lags would require deeper unrolling and complicate the identification algorithm.

**Reference:** `docs/modeling/assumptions.md` (A3, A3a)

### 2. Reflective Measurement Only

| Paper | Framework |
|-------|-----------|
| No restriction on measurement direction | Construct → Indicator only (A1) |

**Rationale:** Formative measurement (indicator → construct) creates identification problems and doesn't fit the LLM-driven extraction paradigm where constructs are theoretical and indicators are operationalizations.

**Reference:** `docs/modeling/scope.md`

### 3. Explicit Confounders (DAGs) vs. ADMGs

| Paper | Framework |
|-------|-----------|
| Bidirected edges (X ↔ Y) for latent confounding | Explicit U nodes (U → X, U → Y) |

**Rationale:** ADMGs obscure the causal structure. Explicit confounders are more interpretable for LLM-proposed models and align with Pearl's SCM formulation. ADMGs are used **internally only** when calling y0's identification algorithm.

**Reference:** `CLAUDE.md` instruction on ADMGs

### 4. No Cross-Classified Models

| Paper | Framework |
|-------|-----------|
| Full cross-classified with Y_{3,t} | Two-level only |

**Rationale:** Cross-classified models require time scales aligned across all individuals (e.g., "day 5 of treatment" means the same for everyone). Most personal data (diaries, wearables) don't have this property.

### 5. No ARMA/ARIMA

| Paper | Framework |
|-------|-----------|
| MA terms via latent variables, ARIMA via fixed coefficients | Not supported |

**Rationale:** Moving average components violate the first-order Markov assumption. The AR(1) restriction subsumes this.

### 6. No TVEM (Time-Varying Effects)

| Paper | Framework |
|-------|-----------|
| Parameters can vary over time: μ_t = g(t) | Time-invariant parameters |

**Rationale:** Focus is on steady-state causal effects, not developmental trajectories. Growth models and time-varying coefficients are out of scope.

---

## Missing Features (Potential Gaps)

These features from the paper are **not currently implemented** but may be valuable additions.

### 1. RDSEM (Residual DSEM)

**Paper's formulation:**
```
Y_it = BX_it + ε̂_it           # structural part (lag-0 only)
ε̂_it = ρε̂_{i,t-1} + ζ_it      # autoregressive residuals
```

**Why it matters:**
- Structural coefficients have same interpretation as cross-sectional regression
- Structural part is **time-interval invariant** (choice of δ doesn't affect it)
- Cleaner separation of "what predicts what" vs. "temporal dynamics"

**Current status:** Not implemented. Framework applies AR on the outcome directly.

**Priority:** Medium - valuable for interpretability

### 2. MEAR(1) / Measurement Error AR(1)

**Paper's formulation:**
```
Y_t = μ + f_t + ε_t       # observed with measurement error
f_t = ϕf_{t-1} + ζ_t      # latent follows AR(1)
```

Equivalent to ARMA(1,1) under parameter constraints. Separates:
- **Measurement error variance** (σ_ε): noise in observation
- **Innovation variance** (σ_ζ): true process variability

**Why it matters:** In social science, measurement error is ubiquitous. Conflating it with innovation variance biases AR coefficient estimates and inflates apparent "inertia."

**Current status:** Framework has measurement models (construct → indicators) but doesn't explicitly model this variance decomposition at the construct level.

**Priority:** High - affects interpretation of AR coefficients

### 3. Latent Centering (Nickell's Bias)

**The problem:** In panel AR models, centering by **sample mean** produces biased AR estimates:
```
Bias ≈ -(1 + ϕ) / (T - 1)
```

For ϕ = 0.3 and T = 10: bias = -0.144 (estimate of 0.16 instead of 0.30).

**Paper's solution:** Center by **latent mean** (Y_{2,i}), not sample mean. The paper's approach handles this automatically because Y_{2,i} is estimated jointly with AR parameters.

**Current status:** Framework uses latent constructs, but unclear if the SSM model builder correctly implements latent centering. Need to verify the AR specification in `SSMModelBuilder`.

**Priority:** Critical - affects validity of all AR estimates

### 4. Random Residual Variances

**Paper's formulation:**
```
V_i = exp(s_{2,i})    # person-specific variance (log-normal)
```

**Why it matters:** Paper's simulation (Table 3) shows that ignoring random variance when correlated with AR coefficient causes:
- Bias in E(ϕ): 0.04 (high correlation case)
- Coverage drops: 97% → 35%

The AR parameter and residual variance are mechanically related via:
```
Var(Y_it | i) = σ² / (1 - ϕ²)
```

**Current status:** SSMSpec supports variance parameters but unclear if person-specific random variances are implemented.

**Priority:** Medium-High - affects AR coefficient estimates when variance heterogeneity exists

### 5. Direct vs. Indirect Covariate Effects

**Paper's three models:**

| Model | Equation | Interpretation |
|-------|----------|----------------|
| Direct | Y = μ + f + β₁X | X affects only concurrent Y |
| Indirect | f_t = ϕf_{t-1} + β₂X | X accumulates through AR process |
| Full | Both | Direct + accumulated effects |

**Why it matters:** Paper shows (Tables 9-10) that misspecifying as direct-only or indirect-only when truth is "full" produces substantial bias in both β and ϕ estimates.

**Current status:** Framework has causal edges but doesn't explicitly distinguish these effect types. An edge X → Y could be either direct or indirect in the paper's sense.

**Priority:** Medium - important for correct causal interpretation

### 6. Unequal Time Intervals

**Paper's approach:** Discretization algorithm (Appendix A) that:
1. Rescales continuous time to integer grid with interval δ
2. Handles subject-specific observation times
3. Inserts missing values between observations
4. Optimal δ balances approximation accuracy vs. missing data proportion

**Why it matters:** ESM/EMA data often have irregular timing. Treating observations as equally-spaced when they're not biases AR estimates.

**Current status:** The CT-SDE formulation handles irregular intervals natively — discretization computes per-interval (Ad, Qd, cd) via matrix exponential, so observations can be arbitrarily spaced without approximation. No additional implementation needed.

**Priority:** Resolved — handled by CT formulation

---

## Ambiguous / Needs Verification

| Feature | Paper | Framework Status | Action Needed |
|---------|-------|------------------|---------------|
| Random slopes (person-specific β_i) | Full support | SSMSpec supports per-subject parameters but unclear if random by person | Verify NumPyro implementation |
| Categorical outcomes | Probit link, thresholds | `measurement_dtype` includes binary/ordinal | Verify NumPyro handles these |
| Model comparison | DIC with detailed caveats | Deferred to ArviZ | Document recommended approach (LOO/WAIC) |
| Dynamic factor analysis | DAFS, WNFS, hybrid models | Latent constructs exist | Out of scope? |

---

## Summary Table

| Feature | Paper | Framework | Status |
|---------|-------|-----------|--------|
| AR(L) for L > 1 | Yes | No | **Intentional** (A3) |
| ARMA/ARIMA | Yes | No | **Intentional** (A3) |
| Cross-classified (Y_{3,t}) | Yes | No | **Intentional** |
| TVEM | Yes | No | **Intentional** |
| Explicit confounders | No (ADMG) | Yes (DAG+U) | **Intentional** |
| Reflective measurement only | No | Yes | **Intentional** (A1) |
| RDSEM | Yes | No | **Gap** |
| MEAR(1) | Yes | Implicit? | **Gap** |
| Latent centering | Critical | Unclear | **Verify** |
| Random variances | Yes | Unclear | **Gap** |
| Direct/indirect effects | Explicit | Implicit | **Gap** |
| Unequal intervals | Algorithm | Native (CT-SDE) | **Resolved** |
| Random slopes | Yes | Unclear | **Verify** |
| Categorical outcomes | Yes | Partial | **Verify** |

---

## Recommended Actions

### High Priority
1. **Verify latent centering** in SSMModelBuilder - critical for unbiased AR estimates
2. **Document MEAR(1) consideration** - decide if measurement error vs. innovation variance decomposition is needed

### Medium Priority
3. **Add RDSEM option** - valuable for interpretability when structural effects are of primary interest
4. **Implement random residual variances** - important when variance heterogeneity is expected
5. **Clarify direct vs. indirect effects** in edge semantics

### Low Priority
6. **Document model comparison approach** - recommend LOO-CV or WAIC via ArviZ

---

## References

- Asparouhov, T., Hamaker, E. L., & Muthén, B. (2017). Dynamic Structural Equation Models. *Structural Equation Modeling*.
- Hamaker, E. L., & Grasman, R. P. P. P. (2015). To center or not to center? Investigating inertia with a multilevel autoregressive model. *Frontiers in Psychology*.
- Nickell, S. (1981). Biases in dynamic models with fixed effects. *Econometrica*.
- Jahn et al. (2025). Identification in temporal causal models. arXiv:2504.20172.
