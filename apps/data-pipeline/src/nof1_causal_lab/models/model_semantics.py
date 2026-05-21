"""Deterministic semantic helpers shared across Stage 4 and SSM compilation."""

from __future__ import annotations

from nof1_causal_lab.artifacts.model_spec import DistributionFamily, LinkFunction

_ADDITIVE_LOCATION_POINT_OPERATORS = frozenset({"first", "last"})
_ADDITIVE_LOCATION_INTERVAL_OPERATORS = frozenset({"mean"})
_NONCENTERABLE_SCALAR_FAMILIES = frozenset(
    {
        DistributionFamily.POISSON,
        DistributionFamily.NEGATIVE_BINOMIAL,
        DistributionFamily.BERNOULLI,
        DistributionFamily.GAMMA,
        DistributionFamily.BETA,
    }
)
_LOCATION_FAMILIES = frozenset({DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T})
_THRESHOLD_FAMILIES = frozenset(
    {
        DistributionFamily.ORDERED_LOGISTIC,
        DistributionFamily.CATEGORICAL,
    }
)


def indicator_has_additive_location_support(
    support_kind: str | None,
    summary_operator: str | None,
) -> bool:
    """Whether an indicator semantics target supports additive location shifts."""
    if support_kind == "point":
        return summary_operator in _ADDITIVE_LOCATION_POINT_OPERATORS
    if support_kind == "interval":
        return summary_operator in _ADDITIVE_LOCATION_INTERVAL_OPERATORS
    return False


def should_auto_center_indicator(
    distribution: DistributionFamily | str,
    link: LinkFunction | str,
    support_kind: str | None,
    summary_operator: str | None,
) -> bool:
    """Return whether deterministic additive centering is semantically admissible."""
    return (
        DistributionFamily(distribution) in _LOCATION_FAMILIES
        and LinkFunction(link) == LinkFunction.IDENTITY
        and indicator_has_additive_location_support(support_kind, summary_operator)
    )


def indicator_requires_observation_intercept(
    distribution: DistributionFamily | str,
    link: LinkFunction | str,
    support_kind: str | None,
    summary_operator: str | None,
    *,
    centered: bool,
) -> bool:
    """Return whether a manifest channel needs a free observation intercept."""
    family = DistributionFamily(distribution)
    resolved_link = LinkFunction(link)

    if family in _THRESHOLD_FAMILIES:
        return False

    if family in _NONCENTERABLE_SCALAR_FAMILIES:
        return True

    if family in _LOCATION_FAMILIES and resolved_link == LinkFunction.IDENTITY:
        return indicator_has_additive_location_support(support_kind, summary_operator) and (
            not centered
        )

    return False
