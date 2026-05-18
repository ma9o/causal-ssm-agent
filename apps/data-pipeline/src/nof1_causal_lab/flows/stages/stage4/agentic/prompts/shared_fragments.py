"""Verbatim Stage 4 prompt fragments shared by reducer and megaprompt modes."""

from nof1_causal_lab.distributions import (
    render_dynamic_prior_scale_guidance,
    render_lagged_beta_authored_interval_guidance,
    render_observation_distribution_guidance_bullets,
    render_observation_link_guidance_bullets,
    render_prior_distribution_guidance_bullets,
)

OBSERVATION_DISTRIBUTION_GUIDANCE_BULLETS = render_observation_distribution_guidance_bullets()
OBSERVATION_LINK_GUIDANCE_BULLETS = render_observation_link_guidance_bullets()
PRIOR_DISTRIBUTION_GUIDANCE_BULLETS = render_prior_distribution_guidance_bullets()
DYNAMIC_PRIOR_SCALE_GUIDANCE = render_dynamic_prior_scale_guidance()
LAGGED_BETA_AUTHORED_INTERVAL_GUIDANCE = render_lagged_beta_authored_interval_guidance()

OBSERVATION_DISTRIBUTION_GUIDANCE_SECTION = (
    "## Observation Distribution Guidance\n\n" + OBSERVATION_DISTRIBUTION_GUIDANCE_BULLETS
)
LINK_FUNCTION_RULES_SECTION = (
    "## Link Function Rules\n\n"
    "Most distributions have exactly one valid link (auto-determined). "
    "You only choose when multiple are valid:\n" + OBSERVATION_LINK_GUIDANCE_BULLETS
)
PRIOR_DISTRIBUTION_TYPES_SECTION = (
    "## Prior Distribution Types\n\n" + PRIOR_DISTRIBUTION_GUIDANCE_BULLETS
)
CONTINUOUS_TIME_DYNAMICS_SECTION = "## Continuous-Time Dynamics\n\n" + DYNAMIC_PRIOR_SCALE_GUIDANCE
LAGGED_EFFECT_INTERVAL_GUIDANCE_SECTION = (
    "## Lagged Effect Interval Guidance\n\n" + LAGGED_BETA_AUTHORED_INTERVAL_GUIDANCE
)
INITIAL_STATE_SCALE_DISCIPLINE_SECTION = (
    "## Initial-State Scale Discipline\n\n"
    "- `t0_mean_*` and `t0_sd_*` live on the latent state scale.\n"
    "- Do not set `t0_mean_*` to the raw reference-indicator mean or "
    "`log(mean(indicator))` just because the indicator uses an identity or log link.\n"
    "- Default to weakly informative latent-scale priors such as `Normal(0, 1)` "
    "and `HalfNormal(1)` unless the construct is explicitly identified on an "
    "observed scale."
)

PRIOR_SOURCE_GUIDANCE = """If you include non-empty `sources`, each entry must be an object with this shape:
```json
{{
  "title": "Source title",
  "snippet": "Relevant excerpt supporting the prior",
  "url": "https://example.org/paper",
  "effect_size": "β=0.21",
  "study_interval_days": 7.0
}}
```

Only `title` and `snippet` are required. Do not use raw strings or ad hoc keys such as `citation`, `finding`, `study_type`, or `notes`. If you are unsure, use `"sources": []`. `study_interval_days` belongs inside each source entry; `reference_interval_days` belongs on the prior itself."""
