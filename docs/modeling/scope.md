# Scope

<!--
Previous versions of this document had a confused understanding of scope. Here's the history:

1. ORIGINAL SIN: We had an "Observability" dimension (Observed/Latent) in the construct taxonomy,
   leading to 8 "types" (2 roles × 2 observabilities × 2 temporal statuses).

2. THE CONFUSION: We excluded Types 3, 6, 7, 8 based on "identification concerns." But
   identification is y0's job in Stage 1b, not a schema-level exclusion. We were doing
   y0's job at the wrong layer.

3. THE REALIZATION: "Observed" just means "has ≥1 indicator." But whether a construct has
   indicators doesn't affect what the schema should accept—it affects whether y0 can
   identify your causal effect. That's a Stage 1b concern.

4. THE FIX: Observability is not a schema-level concern. The schema accepts any DAG. 
   y0 checks identification. The only dimensions that matter for the framework's 
   behavior are Role (exogenous/endogenous) and Temporal (time-varying/time-invariant), 
   because these determine AR structure.

5. STATE-SPACE CONCERN: We worried about "unobserved + time-varying + AR" being a state-space
   problem requiring Kalman filters. But this never arises because:
   - If the unobserved construct blocks identification → y0 rejects, never reaches PyMC
   - If it doesn't block identification → we estimate effects through observed paths, 
     we don't estimate the latent state itself
   - If someone wants latent state values → wrong tool, use dynamax/pykalman
-->

This framework models dynamics of time-varying constructs with optional time-invariant covariates. This is a **causal effect estimation** framework.

---

## Ontology

**Constructs** are theoretical entities in the causal model (stress, mood, cognitive load). They live in the structural model.

**Indicators** are observed data (HRV readings, self-report scores, cortisol levels). They live in the measurement model and reflect their parent construct via factor loadings.

---

## Construct Dimensions

Constructs are classified along two dimensions:

| Dimension | Values | Meaning |
|-----------|--------|---------|
| **Role** | Exogenous / Endogenous | Whether construct receives causal edges from other constructs |
| **Temporal** | Time-varying / Time-invariant | Whether construct changes within person over time |

This yields four construct types:

| Role | Temporal | AR Structure | Example |
|------|----------|--------------|---------|
| Exogenous | Time-varying | None (conditioned on) | Weather, day of week |
| Exogenous | Time-invariant | None (conditioned on) | Age, gender, person intercept |
| Endogenous | Time-varying | AR(1) | Mood, stress, sleep quality |
| Endogenous | Time-invariant | None | Single-occasion outcome |

---

## Autoregressive Structure

**Endogenous time-varying constructs** receive AR(1). See assumptions.md A3.

**Indicators** do not receive AR structure. All temporal dependence in indicator series is attributed to the construct's dynamics. Indicator residuals are assumed iid (A8).

**Exogenous constructs** do not receive AR structure—we condition on their values.

---

## Identification

Whether a causal effect is identified depends on the DAG structure and which constructs have indicators. This is checked by y0 in Stage 1b (after measurement model proposal), not enforced at the schema level.

The schema accepts any valid DAG. y0 determines if your target effect can be estimated.

---

## Out of Scope

**Latent state filtering/smoothing.** If you want to estimate the values of a construct that has no indicators, that's a state-space problem. Use `dynamax`, `pykalman`, or similar tools.

This framework estimates **causal effects between constructs**, not latent state trajectories.