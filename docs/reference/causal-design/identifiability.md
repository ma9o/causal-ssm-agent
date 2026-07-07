# CausalDesign: Identifiability

This reference covers the identifiability assumptions used by [Stage 1b](../../pipeline/01b-measurement-structure-identifiability.md): why treatment-outcome identifiability is checked there, how temporal unrolling works, and why the external contract stays as a DAG with explicit latent confounders.

## What Stage 1b Checks

Stage 1b checks whether each treatment-to-outcome effect is causally identifiable under the latent graph and the measurement assumptions.

The unit of checking is one treatment-outcome pair at a time. Non-identifiability of one effect does not affect identifiability of others, because the ID algorithm[^shpitser2006] restricts attention to ancestors of the outcome; additions elsewhere cannot introduce new blocking structures.

## A3a. Latent Confounders Have Bounded Temporal Reach

**Assumption:** Unobserved constructs follow the same [first-order Markov dynamics](../latent-structure/assumptions.md#a3-markov-property-for-temporal-dynamics) as observed constructs. Latent confounding therefore has maximum latency of one timestep.

**Definition:** A latent confounder `U` can create confounding via:

- `U_t` (contemporaneous): `U_t -> X_t`, `U_t -> Y_t`
- `U_{t-1}` (lagged): `U_{t-1} -> X_t`, `U_{t-1} -> Y_t`

but not via `U_{t-2}` or earlier, because `U_{t-1}` d-separates `U_{t-2}` from current effects under [A3](../latent-structure/assumptions.md#a3-markov-property-for-temporal-dynamics).

**Implication:** A two-timestep graph segment suffices for running the ID algorithm—Markov dynamics prevent confounding paths from reaching beyond one lag.

**Justification:** If observed constructs are Markov, it is parsimonious to assume latent constructs are too. Without this assumption, identification becomes undecidable because one would need to consider confounding paths of arbitrary temporal length.

**Theoretical foundation:** Jahn, Karnik, and Schulman (2025)[^jahn2025] show that for periodic causal graphs with width `w` and latency `L`, running the ID algorithm on a segment of size `O(w × L)` suffices to decide identifiability. Under [A3](../latent-structure/assumptions.md#a3-markov-property-for-temporal-dynamics) plus A3a, `L = 1`, so a two-timestep segment suffices.

## A7. Measurement Structure Identification Enables Causal Identification

**Assumption:** Once the measurement structure is identified, constructs can be treated as effectively observed for the purpose of causal identification via the latent structure.

**Depends on:** [A1](../measurement-structure/assumptions.md#a1-reflective-measurement-structure) (reflective measurement), [A6](../measurement-structure/assumptions.md#a6-measurement-error-handling-depends-on-indicator-count) (multi-indicator identification via factor analysis), and [A9](../measurement-structure/assumptions.md#a9-single-indicator-constructs-absorb-measurement-error) (single-indicator identification by assumption).

**Rationale:** Under A1, A6, and A9 the construct covariance matrix is identified from observed indicator data. Pearl-style[^pearl2009] identification criteria then apply to the construct-level DAG as if constructs were directly observed. This two-step logic—identify the measurement structure, then identify the causal model—follows Anderson and Gerbing (1988)[^anderson1988] and is one reason `CausalDesign`, not `LatentStructure` alone, is the right handoff object for downstream fitting. Proxy indicators for latent confounders follow Miao, Geng, and Tchetgen Tchetgen (2018)[^miao2018].

## User-Facing DAG vs Internal ADMG Projection

The user-facing contract remains a DAG with explicit latent confounder nodes.

Internally, Stage 1b may:

1. unroll that DAG to two timesteps
2. project the unrolled structure to an ADMG
3. run y0's identification algorithm on the internal projection

That projection is an implementation detail. It does not change the external representation.

**Note on latent AR dynamics:** The identification literature is often expressed in ADMG terms[^richardson2002], where latent confounders appear as bidirected edges after projection. That is an internal representation only. The external contract stays as a DAG with explicit latent confounder nodes. The internal dynamics of latent confounders are still not the key object for identification; what matters is which observed variables they jointly affect at which timesteps.

[^pearl2009]: Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press. [Bibliography entry](../bibliography.md)
[^shpitser2006]: Shpitser, I., & Pearl, J. (2006). Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models. *AAAI*, 1955–1960. [Bibliography entry](../bibliography.md)
[^jahn2025]: Jahn, E., Karnik, K., & Schulman, L. J. (2025). Causal Identification in Time Series Models. arXiv:2504.20172. [Bibliography entry](../bibliography.md)
[^anderson1988]: Anderson, J. C., & Gerbing, D. W. (1988). Structural Equation Modeling in Practice. *Psychological Bulletin*, 103(3), 411–423. [Bibliography entry](../bibliography.md)
[^miao2018]: Miao, W., Geng, Z., & Tchetgen Tchetgen, E. J. (2018). Identifying Causal Effects with Proxy Variables. *Biometrika*, 105(4), 987–993. [Bibliography entry](../bibliography.md)
[^richardson2002]: Richardson, T., & Spirtes, P. (2002). Ancestral Graph Markov Models. *Annals of Statistics*, 30(4), 962–1030. [Bibliography entry](../bibliography.md)
