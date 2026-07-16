# Inference Routing for State-Space Models

The implemented inference surface has one method: `marginal_particle_gibbs`. It
targets the joint latent-state and parameter posterior using the true nonlinear
drift, Euler-Maruyama transitions, and the true emission density. For the CT-SDE
formulation, see [estimation.md](estimation.md).

## The Marginalization Challenge

Given latent states **x**_1:T and observations **y**_1:T, parameter inference
would otherwise require the marginal likelihood:

```text
p(y_1:T | theta) = integral p(y_1:T, x_1:T | theta) dx_1:T
```

For T timesteps and n latent dimensions, this integral is over an `(n x T)`
dimensional space. `marginal_particle_gibbs` avoids replacing that integral with
a reported Gaussian approximation. It alternates an exact-invariant dSMC
latent-trajectory update with a parameter move against the directly evaluable
joint posterior.

## Method Taxonomy

| Method | Latent update | Parameter update | Primary use |
|---|---|---|---|
| `marginal_particle_gibbs` | Divide-and-conquer SMC with an exactly corrected leaf proposal | Pseudo-Langevin or random walk | Production fitting |

## Structural Routing

The route always resolves to `marginal_particle_gibbs`. The runtime may build an
IEKS/Laplace backend for initialization, proposal construction, and diagnostics,
but that approximation does not replace the particle posterior target or any
reported simulation path.

## User Overrides

The method itself is fixed. These options tune its proposals without changing
the posterior target:

| Need | Override | Effect |
|---|---|---|
| Select the dSMC leaf proposal | `dsmc_leaf_proposal="amala_exact"` or `dsmc_leaf_proposal="paid_mix"` | Uses exactly corrected aMALA, or a corrected mixture that includes an IEKS-derived pilot component |
| Restrict the latent update | `latent_block_coords=<positive integer>` | Proposes only that many latent coordinates per update; `None` updates all coordinates |
| Select the parameter move | `parameter_proposal="pseudo_langevin"` or `parameter_proposal="random_walk"` | Uses a conditional-gradient drift or a gradient-free random walk |
| Select initialization | `init_method="pathfinder"` or `init_method="random"` | Uses data-informed Pathfinder initialization or random initialization |

## Method Reference

### Marginalized Particle Gibbs

`marginal_particle_gibbs` conditions dSMC on the retained reference trajectory,
then alternates that latent update with a parameter ensemble move.

The only `latent_smoother` value is `"dsmc"`. Its leaf proposal is one of:

- `amala_exact` — gradient-informed aMALA with an auxiliary-potential correction
  that preserves the exact target.

- `paid_mix` — a corrected mixture of the exact aMALA component and pilot-based
  components. IEKS moments shape the proposal only; the importance correction
  preserves the true nonlinear target.

The parameter proposal is `pseudo_langevin` by default. `random_walk` remains
available for gradient-free comparisons.

All posterior samples, diagnostics, posterior-predictive checks, and forward
simulations use the exact production engines. Linearized Gaussian machinery is
limited to initialization or corrected proposal construction.

## Proof-Carrying Result Types

Inference paths have nominally distinct outputs. Laplace/IEKS code returns a
`WarmupProposal`; marginalized Particle Gibbs returns a
`ParticleMCMCPosterior` carrying evidence for the particle engine and nonlinear
Euler-Maruyama transition target. Persisted `FittedArtifact` values accept only
the latter.

Before numeric causal effects are computed, the analysis boundary validates the
persisted engine evidence and joins it to an estimand-specific
`IdentifiedEstimand`. Both proofs carry the same workspace-local causal-design
version. The resulting `CertifiedCausalAnalysis` is the only input accepted by
baseline intervention reporting; interactive simulations construct the same
proof object when loading their artifact context. A warmup result, an
unidentified treatment, or evidence from a different design therefore fails
before numeric reporting.
