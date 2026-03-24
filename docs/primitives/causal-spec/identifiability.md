# CausalSpec: Identifiability

This page owns the identification semantics attached to `CausalSpec`.

## What Stage 1b Checks

Stage 1b checks whether each treatment-to-outcome effect is causally identifiable under the latent graph and the measurement assumptions.

The unit of checking is one treatment-outcome pair at a time. Non-identifiability of one effect does not affect identifiability of others, because the ID algorithm of Shpitser and Pearl (2006) restricts attention to ancestors of the outcome; additions elsewhere cannot introduce new blocking structures.

## A3a. Latent Confounders Have Bounded Temporal Reach

**Assumption:** Unobserved constructs follow the same first-order Markov dynamics as observed constructs. Latent confounding has maximum latency of one timestep.

**Definition:** A latent confounder `U` can create confounding via:

- `U_t` (contemporaneous): `U_t -> X_t`, `U_t -> Y_t`
- `U_{t-1}` (lagged): `U_{t-1} -> X_t`, `U_{t-1} -> Y_t`

but not via `U_{t-2}` or earlier, because `U_{t-1}` d-separates `U_{t-2}` from current effects under the Markov property.

**Implications:**

- A two-timestep graph segment suffices for identification checking
- Confounding "memory" is bounded by the Markov property
- There is no need to reason about arbitrarily long confounding paths through time
- The ID algorithm can be applied to a finite unrolled graph

**Justification:** This is the natural extension of [A3](../latent-model/assumptions.md#a3-markov-property-for-temporal-dynamics) to unobserved constructs. If observed constructs are Markov, it is parsimonious to assume latent constructs are too. Without this assumption, identification becomes undecidable because one would need to consider confounding paths of arbitrary temporal length.

**Theoretical foundation:** Jahn, Karnik, and Schulman (2025) show that for periodic causal graphs with width `w` and latency `L`, running the ID algorithm on a segment of size `O(w × L)` suffices to decide identifiability. Under A3 plus A3a, `L = 1`, so a two-timestep segment suffices.

**Reference:** Jahn, E., Karnik, K., & Schulman, L. J. (2025). Causal Identification in Time Series Models. arXiv:2504.20172.

## A7. Measurement Model Identification Enables Causal Identification

**Assumption:** Once the measurement model is identified, via factor analysis for multi-indicator constructs or by assumption for single-indicator constructs, constructs can be treated as effectively observed for the purpose of causal identification via the latent model.

**Rationale:** Under the pure-indicators assumption, meaning no direct indicator-to-indicator edges, the construct covariance matrix becomes identified from observed indicator covariances. That matrix then serves as "data" for the latent model, and Pearl-style identification criteria apply to the construct-level DAG. This is the logic emphasized in latent-variable SEM treatments such as Anderson and Gerbing (1988) and is one reason `CausalSpec`, not `LatentModel` alone, is the right handoff object for downstream fitting.

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
