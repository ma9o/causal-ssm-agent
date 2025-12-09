# DSEM Specification v1

## Scope

This framework models dynamics of **observed** time-varying constructs with optional time-invariant covariates and random effects. All time-varying constructs must be directly measured. The framework does not support latent state-space models (see Exclusions).

---

## Variable Taxonomy

Variables are classified along three dimensions:

| Dimension | Values | Meaning |
|-----------|--------|---------|
| Role | Endogenous / Exogenous | Whether variable receives causal edges from system |
| Observability | Observed / Latent | Whether variable is directly measured |
| Temporal status | Time-varying / Time-invariant | Whether variable changes within person over time |

### Supported Combinations

| Type | Role | Observability | Temporal | Example | Use |
|------|------|---------------|----------|---------|-----|
| 1 | Exogenous | Observed | Time-varying | Weather, day of week, market index | External time-varying inputs |
| 2 | Exogenous | Observed | Time-invariant | Age, gender, treatment arm | Between-person covariates |
| 4 | Exogenous | Latent | Time-invariant | Person-specific intercept | Random effects for heterogeneity |
| 5 | Endogenous | Observed | Time-varying | Daily mood, sleep quality | Core dynamic system |

---

## Justifications for Supported Types

### Type 1: Exogenous, Observed, Time-varying

External inputs affecting but not affected by the system. No autoregressive structure needed—we condition on observed values. Autocorrelation in exogenous variables exists but is irrelevant since we are not modeling their causes.

### Type 2: Exogenous, Observed, Time-invariant

Between-person covariates. Can predict level-2 intercepts (e.g., "older participants have lower average sleep quality") or moderate within-person dynamics (e.g., "the stress→sleep coefficient is stronger for high-neuroticism individuals").

### Type 4: Exogenous, Latent, Time-invariant

Random effects capturing stable between-person heterogeneity. Identified by repeated observations within person—no indicators required. Partitions variance into between-person and within-person components. Not a causal node; a variance decomposition device that prevents conflation of between and within effects.

### Type 5: Endogenous, Observed, Time-varying

The core use case. Variables with dynamic structure: autoregressive inertia, cross-lagged effects, contemporaneous correlations. Directly observed, well-identified.

---

## Exclusions and Justifications

### Type 3: Exogenous, Latent, Time-varying

An unobserved external shock varying over time. Excluded because identification requires either indicators (making it a factor) or strong structural assumptions. If such a variable is theorized, model it via observed proxy with acknowledged measurement error.

### Type 6: Endogenous, Observed, Time-invariant

A single-occasion outcome. Not a dynamic modeling problem—use standard SEM. Mixing paradigms adds complexity without benefit.

### Type 7: Endogenous, Latent, Time-varying

A latent state with its own dynamics (state-space / Kalman filter territory). Excluded because proper specification requires:

- Prior on initial state distribution
- Prior on process variance (latent state evolution noise)
- Prior on measurement variance (observation noise)

These parameters interact non-trivially. Process and measurement variance are notoriously difficult to disentangle without either multiple indicators or strong domain-informed priors. An automated framework cannot reliably specify these. Users requiring latent dynamics should use specialized state-space tools (e.g., PyMC, Stan) with expert-specified priors.

### Type 8: Endogenous, Latent, Time-invariant

A person-level latent outcome caused by system dynamics. Double identification problem: latent and single observation per person. Not identified without indicators.

---

## Autoregressive Structure

### Rule

All endogenous time-varying variables receive AR(1) at their native timescale by default.

### Justification (Markov Property)

Under the Markov assumption, the state at t-1 is a sufficient statistic for all prior history. Conditioning on Y_{t-1} renders Y_{t-2}, Y_{t-3}, ... conditionally independent of Y_t. Therefore:

- AR(1) captures the relevant temporal dependence
- Higher-order lags add parameters without explanatory benefit under Markovian dynamics
- If residual autocorrelation persists, this suggests missing cross-lags or unmeasured confounders—not higher-order AR

### Cost Asymmetry

Including unnecessary AR(1) wastes one parameter (coefficient ≈ 0, harmless). Omitting necessary AR(1) biases standard errors and inflates cross-lag estimates (harmful). Default inclusion is the conservative choice.

### Exogenous Variables

No AR structure modeled. We condition on observed values; their temporal structure is irrelevant to the causal model.

---

## Temporal Granularity

Variables have an associated time granularity: `hourly`, `daily`, `weekly`, `monthly`, `yearly`, or `None` (time-invariant).

### Model Clock

The model operates at the finest endogenous outcome granularity. If the finest endogenous variable is daily, the model's time index is daily.

### Aggregation at Dimension Level

Raw data may be finer-grained than the dimension's specified granularity. The orchestrator specifies an aggregation function defining how raw observations collapse to the dimension's timescale. Different aggregations encode different substantive meanings:

- Mean: average level matters
- Sum: cumulative amount matters
- Max/Min: extremes matter
- Last: most recent state matters
- Variance: instability itself matters
- Custom: domain-specific aggregations (rolling means, exponential decay, quantiles, etc.)

---

## Cross-Timescale Rules

### Same-Timescale Edges

Two valid lag values under the Markov property:

- **Lag = 0:** Contemporaneous effect within the same time index
- **Lag = 1 granularity unit:** Lagged effect from t-1 to t

Higher-order lags (t-2, t-3, ...) are not permitted. Under Markovian dynamics, t-1 is a sufficient statistic for all prior history. Information from t-2 is already propagated through the AR(1) path.

### Cross-Timescale Edges

**Contemporaneous edges (lag=0) are prohibited.** "Simultaneous" is undefined when variables operate at different grains.

### Coarser Cause → Finer Effect

Lag must equal exactly one unit of the coarser variable's granularity.

**Justification (Markov property):** The AR(1) structure on the coarser variable means its value at t-1 is a sufficient statistic for prior history. Reaching back further is redundant—that information is already propagated through the coarser variable's own autoregressive path.

**Example:** Weekly stress → daily mood requires lag = 168 hours (one week). Last week's stress affects this week's daily mood. Stress from two weeks ago affects last week's stress, which affects this week—the effect is mediated, not direct.

### Finer Cause → Coarser Effect

Requires an aggregation function specifying how fine-grained observations collapse to the coarser outcome's timescale. The orchestrator specifies a Python callable that takes fine-grained values and returns the aggregated predictor.

---

## Schema

```python
from typing import Callable
from pydantic import BaseModel, Field


class Dimension(BaseModel):
    """A variable in the causal model."""

    name: str = Field(description="Variable name (e.g., 'sleep_quality')")
    description: str = Field(description="What this variable represents")
    time_granularity: str | None = Field(
        description="'hourly', 'daily', 'weekly', 'monthly', 'yearly', or None for time-invariant"
    )
    dtype: str = Field(description="'continuous', 'binary', 'ordinal', 'categorical'")
    role: str = Field(description="'endogenous' or 'exogenous'")
    is_latent: bool = Field(
        default=False,
        description="True for random effects. Only valid when role='exogenous' and time_granularity=None"
    )
    aggregation: Callable | None = Field(
        default=None,
        description=(
            "Function to aggregate raw data to this dimension's granularity. "
            "Python callable: takes array of fine-grained values, returns aggregated value. "
            "Required when raw data is finer than time_granularity."
        )
    )


class CausalEdge(BaseModel):
    """A directed causal edge between variables."""

    cause: str = Field(description="Name of cause variable")
    effect: str = Field(description="Name of effect variable")
    lag: int = Field(
        description=(
            "Lag in hours. "
            "Same timescale: 0 (contemporaneous) or exactly 1 granularity unit (Markov property). "
            "Cross-timescale: must equal exactly the coarser variable's granularity in hours."
        )
    )
    aggregation: Callable | None = Field(
        default=None,
        description=(
            "Required when cause is finer-grained than effect. "
            "Python callable: takes array of fine-grained values, returns aggregated value."
        )
    )


class DSEMStructure(BaseModel):
    """Complete DSEM specification."""

    dimensions: list[Dimension] = Field(description="Variables in the model")
    edges: list[CausalEdge] = Field(description="Causal edges including cross-lags")
```

---

## Validation Rules

The following must hold for a valid specification:

1. **Latent validity:** If `is_latent=True`, then `role='exogenous'` and `time_granularity=None`
2. **Endogenous requires time-varying:** If `role='endogenous'`, then `time_granularity` must not be `None`
3. **No inbound edges to exogenous:** If `role='exogenous'`, variable cannot appear as `effect` in any edge
4. **Contemporaneous same-scale only:** If `lag=0`, cause and effect must have identical `time_granularity`
5. **Same-scale lag constraint (Markov):** If cause and effect have identical `time_granularity` and `lag > 0`, lag must equal exactly one granularity unit in hours
6. **Cross-scale lag constraint (Markov):** If cause and effect have different `time_granularity`, `lag` must equal exactly `max(cause_granularity, effect_granularity)` in hours
7. **Aggregation requirement (edges):** If cause `time_granularity` is finer than effect `time_granularity`, `aggregation` must be specified
8. **Aggregation prohibition (edges):** If cause `time_granularity` is coarser or equal to effect `time_granularity`, `aggregation` must be `None`

---

## Interpretation Guidance

Effects are estimated as relationships between **observed** quantities. Measurement error is absorbed into residual variance. Interpret:

- AR coefficients as inertia in the measured variable
- Cross-lag coefficients as predictive relationships between measured variables
- Random effects as stable between-person differences in measured baselines

Do not interpret as effects on "true" latent constructs unless measurement properties are well-established and error is known to be small.
