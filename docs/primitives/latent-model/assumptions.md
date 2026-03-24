# LatentModel Assumptions

This page owns the assumptions whose primary job is to constrain valid construct-level causal structure.

## A3. Markov Property for Temporal Dynamics

**Assumption:** All endogenous time-varying constructs follow first-order Markov dynamics. The state at `t - 1` is a sufficient statistic for all prior history.

**Definition:** For any endogenous construct `C`:

```text
P(C_t | C_t-1, C_t-2, ..., C_1) = P(C_t | C_t-1)
```

**Implications:**

- AR(1) captures the relevant temporal dependence for constructs
- Higher-order lags (AR(2), AR(3), and so on) are not modeled
- Cross-lagged effects use lag-1 at the native granularity
- Residual autocorrelation suggests missing cross-lags or unmeasured confounders, not higher-order AR

**Justification:** The Markov property is a parsimony constraint with asymmetric costs. Including unnecessary AR(1) wastes one parameter (coefficient approximately 0, harmless). Omitting necessary AR(1) biases standard errors and inflates cross-lag estimates (harmful). Default AR(1) is the conservative choice.

## A4. Acyclicity Within Time Slice

**Assumption:** Contemporaneous causal relationships within the same time index must form a directed acyclic graph.

**Definition:** If we consider only edges where lag = 0, the resulting graph must have no cycles.

**Implications:**

- Feedback loops must be modeled via lagged edges across time
- Contemporaneous relationships represent instantaneous causation or common response to unmodeled causes
- Standard DAG-based identification algorithms apply within each time slice

**Justification:** Cyclic contemporaneous relationships are not identified without additional constraints such as instrumental variables or non-Gaussianity. Requiring acyclicity simplifies identification while allowing feedback dynamics through the temporal structure, as in dynamic SEM treatments such as Asparouhov, Hamaker, and Muthen (2018).

**Identification implication:** When checking causal identifiability, Stage 1b unrolls the temporal graph to two timesteps and applies the Shpitser-Pearl ID algorithm to that finite graph. This correctly handles lagged confounding; an unobserved `U_{t-1}` affecting both `X_t` and `Y_t` blocks identification of `X_t -> Y_t`. The unrolled DAG is projected internally to an ADMG for the ID algorithm.

**References:**

- Asparouhov, T., Hamaker, E. L., & Muthen, B. (2018). Dynamic structural equation models. *Structural Equation Modeling*, 25(3), 359-388. https://doi.org/10.1080/10705511.2017.1406803
- Shpitser, I., & Pearl, J. (2006). *Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models.* AAAI.

## A5. Time-Invariant Latents as Subject-Level Static States

**Assumption:** Time-invariant constructs capture stable subject-level differences over the modeled window. They may be exogenous or endogenous, but any modeled causes of a time-invariant construct must themselves be time-invariant.

**Definition:** A time-invariant latent is implemented downstream as a quasi-constant state: its drift diagonal is set to approximately 0 and its diffusion to approximately 0, so `eta_i(t) ≈ eta_i(0)` throughout the time series. The semantic commitment starts here even though the exact implementation detail belongs to the runtime.

**Implications:**

- Time-invariant latents act as subject-specific static states, absorbing stable baseline differences
- They may serve as stable covariates or as static outcomes or traits explained by other stable constructs
- They cannot have time-varying parents, because a within-person changing cause cannot determine a within-person fixed child
- If a time-invariant construct has parents, those parents must also be time-invariant
- They affect time-varying constructs at every timestep after temporal unrolling

**Note on hierarchical modeling:** The current implementation fits each subject independently. There is no cross-subject shrinkage or hierarchical prior on these intercepts. True random effects in the multilevel sense would require a hierarchical model where subject-level parameters are drawn from a population distribution. This is not currently supported.

**Justification:** In intensive longitudinal data, ignoring stable individual differences biases within-person effect estimates. Static subject-level states are the minimal adjustment for this confound. Allowing time-invariant constructs to depend on other time-invariant constructs still preserves that interpretation while ruling out incoherent within-person arrows into fixed variables.

## Boundary

A3, A4, and A5 are primarily about what the construct-level graph is allowed to say. Identification-specific assumptions that use the graph live with the [CausalSpec](../causal-spec/identifiability.md).
