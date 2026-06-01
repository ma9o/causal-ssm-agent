# Inference Routing for State-Space Models

The implemented inference surface is deliberately small: a single default method, `marginal_particle_gibbs`, plus `particle_marginal_mh` as a pseudo-marginal comparator. For the CT-SDE formulation and likelihood backends, see [estimation.md](estimation.md).

## The Marginalization Challenge

Given a state-space model with latent states **x**_1:T and observations **y**_1:T, parameter inference requires the marginal likelihood:

```text
p(y_1:T | theta) = integral p(y_1:T, x_1:T | theta) dx_1:T
```

For SSMs with T timesteps and n latent dimensions, this integral is over an `(n x T)`-dimensional space. `marginal_particle_gibbs` avoids forming it directly: it targets the directly evaluable joint latent/parameter posterior with a collapsed Particle Gibbs sweep, alternating a conditional-SMC latent-trajectory update with a parameter move. `particle_marginal_mh` instead targets the parameter posterior alone, using a bootstrap particle filter to produce an unbiased estimate of the marginal likelihood inside a pseudo-marginal Metropolis-Hastings accept/reject.

## Method Taxonomy

| Method | Strategy | Latent update | Parameter update | Primary use |
|---|---|---|---|---|
| `marginal_particle_gibbs` | Collapsed Particle Gibbs over the joint latent/parameter posterior | Conditional SMC smoother (selectable: `plain`, `amala`, `amala_plus`, `mgrad`, `dsmc`) | Pseudo-Langevin or random-walk parameter proposal | Default fit |
| `particle_marginal_mh` | Pseudo-marginal MH over parameters | Bootstrap particle-filter marginal-likelihood estimate; latent paths not retained | Preconditioned Metropolis-Hastings | Comparator / validation |

## Structural Routing

The default routing resolves to `marginal_particle_gibbs`. Routing also records the structural backend for runtime/frontend diagnostics:

- **`laplace`**: non-Gaussian observations or support-aware summaries use the IEKS/Laplace approximate marginal likelihood when a method needs a marginal-likelihood objective.

## User Overrides

Selection happens on two axes. The method is chosen with the `method` argument to [`inference.fit()`](estimation.md#data-flow); within `marginal_particle_gibbs` the latent smoother and parameter proposal are chosen with keyword arguments.

| Need | Override | Why |
|---|---|---|
| Pseudo-marginal parameter inference with a bootstrap likelihood | `method="particle_marginal_mh"` | Targets the parameter posterior alone; an independent check on the collapsed sampler. |
| A different latent-trajectory smoother | `latent_smoother=` `"plain"` \| `"amala"` \| `"amala_plus"` \| `"mgrad"` \| `"dsmc"` | Trades off mixing, gradient use, and parallel-in-time depth for the conditional-SMC latent update. |
| A gradient-informed vs. gradient-free parameter move | `parameter_proposal=` `"pseudo_langevin"` (default) \| `"random_walk"` | Pseudo-Langevin uses a conditional parameter-gradient drift; random-walk is the gradient-free fallback. |

## Runtime Conditioning

The runtime contains Polya-Gamma augmentation and Rao-Blackwellized particle filtering (RBPF) machinery, but neither is reachable through the current inference surface: both `marginal_particle_gibbs` and `particle_marginal_mh` require `enable_polya_gamma=False` and `rbpf_mode="none"` and raise otherwise. The collapsed Particle Gibbs target and the pseudo-marginal target are defined without those augmentations, so they are rejected explicitly rather than silently ignored.

## Method Reference

### Marginalized Particle Gibbs

`marginal_particle_gibbs` (default) targets the directly evaluable latent/parameter posterior with a collapsed Particle Gibbs update: a conditional-SMC latent-trajectory smoother conditioned on the retained reference path, alternated with a parameter move.

**Latent smoothers** (`latent_smoother`):

- `plain` — plain conditional SMC (cSMC) over the latent trajectory. Default.
- `amala` / `amala_plus` — particle-aMALA and particle-aMALA+ smoothers, which add a Metropolis-adjusted Langevin move to the conditional particle update.
- `mgrad` — particle-mGRAD smoother, a prior-informed gradient proposal. Its `latent_kernel_algorithm` selects the sequential `particle_mgrad` construction or the parallel-in-time `pit_aux_csmc` construction.
- `dsmc` — divide-and-conquer SMC smoother with logarithmic parallel depth in T.

**Parameter proposals** (`parameter_proposal`): `pseudo_langevin` (default) uses a conditional parameter-gradient drift; `random_walk` is the gradient-free fallback.

**When to use:** Default complete-data inference. The smoother and proposal knobs tune mixing and parallel-in-time cost without changing the target.

### Particle Marginal Metropolis-Hastings

`particle_marginal_mh` targets the parameter posterior using a bootstrap particle-filter estimate of the marginal likelihood inside a pseudo-marginal MH accept/reject. It shares the discretized runtime bundle with the default method.

**When to use:** As an independent comparator on the parameter posterior.

**Limitations:** Does not retain latent paths or compute latent posterior summaries. Pseudo-marginal mixing degrades if the particle count is too low for the likelihood-estimator variance.
