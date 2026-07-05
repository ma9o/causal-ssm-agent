"""Verbatim Stage 4 prompt fragments shared by the construct-admission prompt."""

from nof1_causal_lab.distributions import (
    render_dynamic_prior_scale_guidance,
    render_observation_distribution_guidance_bullets,
    render_observation_link_guidance_bullets,
    render_prior_distribution_guidance_bullets,
)

OBSERVATION_DISTRIBUTION_GUIDANCE_BULLETS = render_observation_distribution_guidance_bullets()
OBSERVATION_LINK_GUIDANCE_BULLETS = render_observation_link_guidance_bullets()
PRIOR_DISTRIBUTION_GUIDANCE_BULLETS = render_prior_distribution_guidance_bullets()
DYNAMIC_PRIOR_SCALE_GUIDANCE = render_dynamic_prior_scale_guidance()

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
