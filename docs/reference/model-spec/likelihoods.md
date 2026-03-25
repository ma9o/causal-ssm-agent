# Likelihoods

Defines the observation-model vocabulary for [`LikelihoodSpec`](../../pipeline/04-model-specification-priors.md#likelihoodspec) entries in a [`ModelSpec`](../../pipeline/04-model-specification-priors.md#modelspec).

## Dtype-to-Distribution Mapping

Each indicator's [`measurement_dtype`](../../pipeline/01b-measurement-identifiability.md#indicator) determines the default distribution and link function. Where the dtype admits only one valid combination, the likelihood is locked deterministically by the [Stage 4 skeleton](../../pipeline/04-model-specification-priors.md). Where alternatives exist, the LLM chooses via a decision card.

| `measurement_dtype` | Default distribution | Link | Alternatives |
|---|---|---|---|
| `continuous` | `gaussian` | `identity` | `student_t`, `gamma` (`log` or `inverse`), `beta` (`logit` or `probit`) |
| `binary` | `bernoulli` | `logit` | `bernoulli` with `probit` |
| `count` | `poisson` | `log` | `negative_binomial` (`log`) |
| `ordinal` | `ordered_logistic` | `cumulative_logit` | None |
| `categorical` | `categorical` | `softmax` | `ordered_logistic` (`cumulative_logit`) when categories are substantively ordered |

## Distribution Families

`DistributionFamily` enumerates the valid likelihood distribution names: `gaussian`, `student_t`, `gamma`, `beta`, `bernoulli`, `poisson`, `negative_binomial`, `ordered_logistic`, and `categorical`.

## Link Functions

`LinkFunction` enumerates the valid link function names: `identity`, `log`, `inverse`, `logit`, `probit`, `cumulative_logit`, and `softmax`.
