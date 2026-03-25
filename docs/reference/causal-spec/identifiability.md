# CausalSpec: Identifiability

This reference covers the identifiability assumptions used by [Stage 1b](../../pipeline/01b-measurement-identifiability.md): why treatment-outcome identifiability is checked there, how temporal unrolling works, and why the external contract stays as a DAG with explicit latent confounders.

## What Stage 1b Checks

Stage 1b checks whether each treatment-to-outcome effect is causally identifiable under the latent graph and the measurement assumptions.

The unit of checking is one treatment-outcome pair at a time. Non-identifiability of one effect does not affect identifiability of others, because the ID algorithm of Shpitser and Pearl (2006) restricts attention to ancestors of the outcome; additions elsewhere cannot introduce new blocking structures.

## A3a. Latent Confounders Have Bounded Temporal Reach

**Assumption:** Unobserved constructs follow the same [first-order Markov dynamics](../latent-model/assumptions.md#a3-markov-property-for-temporal-dynamics) as observed constructs. Latent confounding therefore has maximum latency of one timestep.

**Definition:** A latent confounder `U` can create confounding via:

- `U_t` (contemporaneous): `U_t -> X_t`, `U_t -> Y_t`
- `U_{t-1}` (lagged): `U_{t-1} -> X_t`, `U_{t-1} -> Y_t`

but not via `U_{t-2}` or earlier, because `U_{t-1}` d-separates `U_{t-2}` from current effects under [A3](../latent-model/assumptions.md#a3-markov-property-for-temporal-dynamics).

**Implication:** A two-timestep graph segment suffices for running the ID algorithm—Markov dynamics prevent confounding paths from reaching beyond one lag.

**Justification:** If observed constructs are Markov, it is parsimonious to assume latent constructs are too. Without this assumption, identification becomes undecidable because one would need to consider confounding paths of arbitrary temporal length.

**Theoretical foundation:** Jahn, Karnik, and Schulman (2025) show that for periodic causal graphs with width `w` and latency `L`, running the ID algorithm on a segment of size `O(w × L)` suffices to decide identifiability. Under [A3](../latent-model/assumptions.md#a3-markov-property-for-temporal-dynamics) plus A3a, `L = 1`, so a two-timestep segment suffices.

**Reference:** Jahn, E., Karnik, K., & Schulman, L. J. (2025). Causal Identification in Time Series Models. arXiv:2504.20172.

## A7. Measurement Model Identification Enables Causal Identification

**Assumption:** Once the measurement model is identified, constructs can be treated as effectively observed for the purpose of causal identification via the latent model.

**Depends on:** [A1](../measurement-model/assumptions.md#a1-reflective-measurement-model) (reflective measurement), [A6](../measurement-model/assumptions.md#a6-measurement-error-handling-depends-on-indicator-count) (multi-indicator identification via factor analysis), and [A9](../measurement-model/assumptions.md#a9-single-indicator-constructs-absorb-measurement-error) (single-indicator identification by assumption).

**Rationale:** Under A1, A6, and A9 the construct covariance matrix is identified from observed indicator data. Pearl-style identification criteria then apply to the construct-level DAG as if constructs were directly observed. This two-step logic—identify the measurement model, then identify the causal model—follows Anderson and Gerbing (1988) and is one reason `CausalSpec`, not `LatentModel` alone, is the right handoff object for downstream fitting.

**References:**

- Anderson, J. C., & Gerbing, D. W. (1988). Structural equation modeling in practice: A review and recommended two-step approach. *Psychological Bulletin*, 103(3), 411-423.
- Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying causal effects with proxy variables of an unmeasured confounder. *Biometrika*, 105(4), 987-993.

## User-Facing DAG vs Internal ADMG Projection

The user-facing contract remains a DAG with explicit latent confounder nodes.

Internally, Stage 1b may:

1. unroll that DAG to two timesteps
2. project the unrolled structure to an ADMG
3. run y0's identification algorithm on the internal projection

That projection is an implementation detail. It does not change the external representation.

**Note on latent AR dynamics:** The identification literature is often expressed in ADMG terms, where latent confounders appear as bidirected edges after projection. That is an internal representation only. The external contract stays as a DAG with explicit latent confounder nodes. The internal dynamics of latent confounders are still not the key object for identification; what matters is which observed variables they jointly affect at which timesteps.

**Reference:** Shpitser, I., & Pearl, J. (2006). *Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models.* AAAI.
