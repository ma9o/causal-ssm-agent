# LatentStructure Assumptions

This page collects the assumptions that constrain valid construct-level causal structure.

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

**Justification:** First-order within-person dynamics are a standard starting point in dynamic SEM[^asparouhov2018]. In this project, the Markov property is also a parsimony constraint with asymmetric costs. Including unnecessary AR(1) wastes one parameter (coefficient approximately 0, harmless). Omitting necessary AR(1) biases standard errors and inflates cross-lag estimates (harmful). Default AR(1) is the conservative choice.

## A4. Acyclicity Within Time Slice

**Assumption:** Contemporaneous causal relationships within the same time index must form a directed acyclic graph.

**Definition:** If we consider only edges where lag = 0, the resulting graph must have no cycles.

**Implications:**

- Feedback loops must be modeled via lagged edges across time
- Contemporaneous relationships represent instantaneous causation or common response to unmodeled causes
- Standard DAG-based identification algorithms apply within each time slice

**Justification:** Cyclic contemporaneous relationships are not identified without additional constraints such as instrumental variables or non-Gaussianity. Requiring acyclicity simplifies identification while allowing feedback dynamics through the temporal structure, as in dynamic SEM treatments[^asparouhov2018].

**Identification implication:** A4 interacts with identifiability checking[^shpitser2006]. See [A3a](../causal-design/identifiability.md#a3a-latent-confounders-have-bounded-temporal-reach) for how Stage 1b unrolls the temporal graph to two timesteps and [the ADMG projection](../causal-design/identifiability.md#user-facing-dag-vs-internal-admg-projection) for the internal ID algorithm.

## A4b. Endogenous Time-Varying Directed Effects Are Drift-Mediated

**Assumption:** Directed effects between endogenous time-varying constructs are represented through continuous-time drift rather than contemporaneous within-slice arrows.

**Definition:** For endogenous time-varying constructs `X` and `Y`, the user-facing `LatentStructure` does not permit a same-slice directed edge `X_t -> Y_t`. Such a relation must be represented as a lagged edge `X_t-1 -> Y_t` in the graph and compiles downstream to a cross-effect in the [continuous-time drift](../estimation.md#1-ct-sde-formulation) (an off-diagonal of the drift matrix in the linear case). Same-time dependence at `t` belongs to explicit confounding structure or shared innovations, not a directed within-slice arrow.

**Implications:**

- Stage 1a rejects `lagged=false` edges between constructs that are both endogenous and time-varying
- Directed cross-construct effects among latent states are encoded as drift-mediated temporal dependence
- Same-time co-movement should be represented by explicit latent confounders or diffusion covariance, depending on whether the dependence is theoretical or stochastic

**Justification:** The chosen latent-process family is a multivariate continuous-time SDE. In that model class, directed coupling between evolving latent states is carried by the drift term, while contemporaneous dependence is captured by common causes or shared innovation. This is therefore a project modeling-contract choice for the current runtime, not a claim that every dynamic latent-variable model must encode same-slice effects this way.

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

**Justification:** In intensive longitudinal data, failing to separate stable between-subject differences from within-person dynamics can bias lagged effect estimates[^hamaker2015]. Static subject-level states are the minimal adjustment for this confound. Allowing time-invariant constructs to depend on other time-invariant constructs still preserves that interpretation while ruling out incoherent within-person arrows into fixed variables.

## Boundary

A3, A4, A4b, and A5 constrain what the construct-level graph is allowed to say. Identification-specific assumptions that use that graph live with the [CausalDesign](../causal-design/identifiability.md).

[^asparouhov2018]: Asparouhov, T., Hamaker, E. L., & Muthén, B. (2018). Dynamic Structural Equation Models. *Structural Equation Modeling*, 25(3), 359–388. [Bibliography entry](../bibliography.md)
[^shpitser2006]: Shpitser, I., & Pearl, J. (2006). Identification of Joint Interventional Distributions in Recursive Semi-Markovian Causal Models. *AAAI*, 1955–1960. [Bibliography entry](../bibliography.md)
[^hamaker2015]: Hamaker, E. L., Kuiper, R. M., & Grasman, R. P. P. P. (2015). A Critique of the Cross-Lagged Panel Model. *Psychological Methods*, 20(1), 102–116. [Bibliography entry](../bibliography.md)
