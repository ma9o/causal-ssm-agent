"""Generate a production-shaped artificial completion of the canonical DEMO fixture.

The checked-in DEMO episode predates executable-disposition declarations and ends
after validation. This script reads that episode's real latent structure,
measurement structure, validation profiles, and panel; selects a compact,
nonredundant scientific story; derives its production causal/structural projection;
and materializes the missing downstream projections needed by Storybook. A future
complete episode promotion replaces these files.

This is presentation data, not a fitted scientific result.  The generated values
are deliberately plausible and internally coherent, but they are never consumed
by the production pipeline.

Usage::

    bun run fixture:complete-demo
    bun run fixture:check-demo
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import statistics
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from nof1_causal_lab.artifacts.causal_design import CausalDesign
from nof1_causal_lab.artifacts.statistical_model_spec import (
    validate_statistical_model_spec_dict,
)
from nof1_causal_lab.flows.transitions.analysis.contracts import (
    BaselineReportContract,
    SimulateScenarioInput,
    SimulateScenarioToolResultContract,
)
from nof1_causal_lab.flows.transitions.inference.contracts import PosteriorContract
from nof1_causal_lab.flows.transitions.measurement_structure.assemble import (
    build_causal_design,
)
from nof1_causal_lab.flows.transitions.measurement_structure.contracts import (
    MeasurementStructureContract,
)
from nof1_causal_lab.flows.transitions.measurement_structure.grounding import (
    measurement_structure_grounding,
)
from nof1_causal_lab.flows.transitions.model_spec.assembly import validate_assembly
from nof1_causal_lab.flows.transitions.model_spec.contracts import (
    StatisticalModelSpecContract,
)
from nof1_causal_lab.flows.transitions.validation.contracts import ValidationReportContract
from nof1_causal_lab.models.structural import build_structural_plan
from nof1_causal_lab.utils.identifiability import check_identifiability
from nof1_causal_lab.utils.structural_plan import (
    get_edges,
    get_manifest_indicators,
    get_state_names,
)

if TYPE_CHECKING:
    from nof1_causal_lab.artifacts.structural_plan import StructuralPlan

JsonObject = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_ROOT = REPO_ROOT / "data" / "DEMO"
ARTIFACT_ROOT = DEMO_ROOT / "fixture" / "artifacts"
TRACE_ROOT = DEMO_ROOT / "fixture" / "traces"
STORE_ROOT = DEMO_ROOT / "store"
PANEL_PATH = DEMO_ROOT / "store" / "panel" / "v1" / "panel.parquet"

HORIZON_DAYS = 60
PPC_POINTS = 61
PRIOR_SAMPLE_COUNT = 96
POSTERIOR_DRAW_COUNT = 192
MARGINAL_POINTS = 51

# The presentation fixture is a deliberately compact scientific projection of the
# much richer stored DEMO proposal.  It keeps the original measurement values and
# semantic definitions, but removes redundant proxy channels and preserves the
# scientifically important mechanisms that the executable model cannot support.
DEMO_CONSTRUCT_ORDER = (
    "natural_recovery_propensity",
    "past_escitalopram_response_tolerability",
    "patient_taper_preference_beliefs",
    "taper_speed_dose_reduction",
    "adherence_to_regimen",
    "escitalopram_dose_taken",
    "duration_current_escitalopram_use",
    "stable_withdrawal_susceptibility",
    "neuroadaptation_dependence_state",
    "withdrawal_symptom_burden",
    "external_stressful_events",
    "perceived_stress_burden",
    "physical_activity",
    "sleep_circadian_disruption",
    "clinical_monitoring_rescue_care",
    "other_psychotropic_or_somatic_treatments",
    "internalizing_symptom_burden",
)

DEMO_INDICATOR_ORDER = (
    "phq9_screening_score",
    "gad7_screening_score",
    "state_of_mind_valence",
    "journal_internalizing_symptom_severity",
    "taper_speed_instruction_intensity",
    "escitalopram_documented_dose_mg",
    "medication_adherence_documented",
    "sleep_efficiency_pct",
    "daily_step_count",
    "journal_stress_severity",
    "external_stressor_event_count",
    "other_treatment_change_count",
    "patient_taper_preference_level",
    "natural_recovery_evidence_level",
    "past_escitalopram_response_tolerability_level",
    "current_escitalopram_use_duration_proxy",
    "long_term_ssri_adaptation_proxy",
    "clinical_contact_count",
    "rescue_or_monitoring_action",
)

# Explicit roots of the reduced scientific boundary. Their omitted upstream
# systems are intentionally outside this compact story rather than silently
# leaving them as unexplained endogenous constructs.
DEMO_CONSTRUCT_ROLE_OVERRIDES = {
    "natural_recovery_propensity": "exogenous",
    "stable_withdrawal_susceptibility": "exogenous",
}

DEMO_KNOWN_INPUTS: tuple[JsonObject, ...] = (
    {
        "construct": "external_stressful_events",
        "source_indicator": "external_stressor_event_count",
        "scale": 1.0,
        "missing_policy": "zero",
    },
    {
        "construct": "other_psychotropic_or_somatic_treatments",
        "source_indicator": "other_treatment_change_count",
        "scale": 1.0,
        "missing_policy": "zero",
    },
)

DEMO_SCIENTIFIC_ONLY: tuple[JsonObject, ...] = (
    {
        "construct": "patient_taper_preference_beliefs",
        "reason": (
            "The sparse semantic proxy is entangled with the documented taper plan, so it "
            "remains assignment context rather than an executable latent state."
        ),
    },
    {
        "construct": "duration_current_escitalopram_use",
        "reason": (
            "Historical exposure duration is observed only as sparse static context and feeds "
            "an unsupported neuroadaptation mechanism."
        ),
    },
    {
        "construct": "neuroadaptation_dependence_state",
        "reason": (
            "The available text-derived adaptation proxy cannot independently resolve the "
            "physiologic state from duration and taper documentation."
        ),
    },
    {
        "construct": "natural_recovery_propensity",
        "reason": (
            "Only sparse retrospective evidence is available for the untreated illness course, "
            "so it remains scientific and identification context."
        ),
    },
    {
        "construct": "past_escitalopram_response_tolerability",
        "reason": (
            "Historical response is represented by a sparse retrospective summary, so it "
            "remains measured assignment context rather than an executable dynamic state."
        ),
    },
    {
        "construct": "clinical_monitoring_rescue_care",
        "reason": (
            "Clinical contacts and rescue actions are sparse, event-triggered assignment "
            "signals that preserve treatment-change context without supporting a latent state."
        ),
    },
)

# Edges are selected from the stored theory by endpoint.  The three explicit
# replacements contract well-understood mechanism chains that had no independent,
# varying measurement support in this episode.
DEMO_EDGE_ORDER = (
    ("natural_recovery_propensity", "internalizing_symptom_burden"),
    ("natural_recovery_propensity", "past_escitalopram_response_tolerability"),
    ("past_escitalopram_response_tolerability", "duration_current_escitalopram_use"),
    ("past_escitalopram_response_tolerability", "patient_taper_preference_beliefs"),
    ("internalizing_symptom_burden", "patient_taper_preference_beliefs"),
    ("patient_taper_preference_beliefs", "taper_speed_dose_reduction"),
    ("patient_taper_preference_beliefs", "adherence_to_regimen"),
    ("taper_speed_dose_reduction", "escitalopram_dose_taken"),
    ("adherence_to_regimen", "escitalopram_dose_taken"),
    ("internalizing_symptom_burden", "adherence_to_regimen"),
    ("withdrawal_symptom_burden", "escitalopram_dose_taken"),
    ("withdrawal_symptom_burden", "patient_taper_preference_beliefs"),
    ("duration_current_escitalopram_use", "neuroadaptation_dependence_state"),
    ("stable_withdrawal_susceptibility", "withdrawal_symptom_burden"),
    ("taper_speed_dose_reduction", "withdrawal_symptom_burden"),
    ("neuroadaptation_dependence_state", "withdrawal_symptom_burden"),
    ("withdrawal_symptom_burden", "sleep_circadian_disruption"),
    ("withdrawal_symptom_burden", "perceived_stress_burden"),
    ("withdrawal_symptom_burden", "internalizing_symptom_burden"),
    ("escitalopram_dose_taken", "internalizing_symptom_burden"),
    ("external_stressful_events", "perceived_stress_burden"),
    ("perceived_stress_burden", "sleep_circadian_disruption"),
    ("perceived_stress_burden", "internalizing_symptom_burden"),
    ("physical_activity", "sleep_circadian_disruption"),
    ("physical_activity", "internalizing_symptom_burden"),
    ("sleep_circadian_disruption", "internalizing_symptom_burden"),
    ("internalizing_symptom_burden", "sleep_circadian_disruption"),
    ("internalizing_symptom_burden", "physical_activity"),
    ("withdrawal_symptom_burden", "clinical_monitoring_rescue_care"),
    ("internalizing_symptom_burden", "clinical_monitoring_rescue_care"),
    ("clinical_monitoring_rescue_care", "other_psychotropic_or_somatic_treatments"),
    ("other_psychotropic_or_somatic_treatments", "internalizing_symptom_burden"),
)

DEMO_REDUCED_EDGES: dict[tuple[str, str], JsonObject] = {
    ("natural_recovery_propensity", "internalizing_symptom_burden"): {
        "cause": "natural_recovery_propensity",
        "effect": "internalizing_symptom_burden",
        "description": (
            "Underlying remission and relapse propensity changes symptom burden independently "
            "of the taper pathway."
        ),
        "lagged": True,
        "sources": [],
    },
    ("taper_speed_dose_reduction", "withdrawal_symptom_burden"): {
        "cause": "taper_speed_dose_reduction",
        "effect": "withdrawal_symptom_burden",
        "description": (
            "Faster or larger dose reductions can precipitate withdrawal when physiologic "
            "adaptation and susceptibility are present."
        ),
        "lagged": True,
        "sources": [],
    },
    ("escitalopram_dose_taken", "internalizing_symptom_burden"): {
        "cause": "escitalopram_dose_taken",
        "effect": "internalizing_symptom_burden",
        "description": (
            "Actual dose changes the maintenance of antidepressant benefit and can alter "
            "subsequent internalizing symptom burden."
        ),
        "lagged": True,
        "sources": [],
    },
}


@dataclass(frozen=True)
class FixtureSources:
    """Current production projection deterministically derived from the stored DEMO episode."""

    latent_artifact: JsonObject
    measurement_artifact: JsonObject
    causal_artifact: JsonObject
    structural_plan_artifact: JsonObject
    validation_artifact: JsonObject
    structural_plan: StructuralPlan
    state_constructs: list[JsonObject]
    executable_edges: list[JsonObject]
    manifest_indicators: list[JsonObject]
    profiles: dict[str, JsonObject]
    simulatable_treatments: list[str]


def _read_json(path: Path) -> JsonObject:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object at {path}")
    return payload


def _json_bytes(payload: JsonObject) -> bytes:
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode()


def _round(value: float, digits: int = 5) -> float:
    return round(float(value), digits)


def _seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")


def _seed_fraction(label: str) -> float:
    return _seed(label) / float(2**64 - 1)


def _rng(label: str) -> random.Random:
    return random.Random(_seed(label))


def _normal_pdf(value: float, mean: float, sd: float) -> float:
    z = (value - mean) / sd
    return math.exp(-0.5 * z * z) / (sd * math.sqrt(2 * math.pi))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count == 1:
        return [start]
    return [start + (stop - start) * index / (count - 1) for index in range(count)]


def _density_points(distribution: str, params: JsonObject) -> list[JsonObject]:
    if distribution == "Normal":
        mean = float(params["mu"])
        sd = float(params["sigma"])
        xs = _linspace(mean - 3.5 * sd, mean + 3.5 * sd, MARGINAL_POINTS)
        ys = [_normal_pdf(value, mean, sd) for value in xs]
    elif distribution == "HalfNormal":
        sd = float(params["sigma"])
        xs = _linspace(0, 4 * sd, MARGINAL_POINTS)
        ys = [2 * _normal_pdf(value, 0, sd) for value in xs]
    elif distribution == "Beta":
        alpha = float(params["alpha"])
        beta = float(params["beta"])
        coefficient = math.exp(math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta))
        xs = _linspace(0.001, 0.999, MARGINAL_POINTS)
        ys = [coefficient * value ** (alpha - 1) * (1 - value) ** (beta - 1) for value in xs]
    elif distribution == "Gamma":
        concentration = float(params["concentration"])
        rate = float(params["rate"])
        mean = concentration / rate
        sd = math.sqrt(concentration) / rate
        coefficient = rate**concentration / math.gamma(concentration)
        xs = _linspace(0.001, mean + 4 * sd, MARGINAL_POINTS)
        ys = [coefficient * value ** (concentration - 1) * math.exp(-rate * value) for value in xs]
    elif distribution == "TruncatedNormal":
        mean = float(params["mu"])
        sd = float(params["sigma"])
        lower = float(params["lower"])
        upper = float(params["upper"])
        normalization = _normal_cdf((upper - mean) / sd) - _normal_cdf((lower - mean) / sd)
        xs = _linspace(lower, upper, MARGINAL_POINTS)
        ys = [_normal_pdf(value, mean, sd) / normalization for value in xs]
    else:
        raise ValueError(f"No fixture density implementation for {distribution}")
    return [{"x": _round(x), "y": _round(y)} for x, y in zip(xs, ys, strict=True)]


def _indicator_standardized(indicator: JsonObject, distribution: str) -> bool:
    if distribution not in {"gaussian", "student_t"}:
        return False
    support = indicator.get("support_kind")
    operator = indicator.get("summary_operator")
    return (support == "point" and operator in {"first", "last"}) or (
        support == "interval" and operator == "mean"
    )


def _likelihood_for(indicator: JsonObject, profile: JsonObject) -> JsonObject:
    dtype = str(indicator["measurement_dtype"])
    if dtype == "binary":
        distribution, link = "bernoulli", "logit"
    elif dtype == "ordinal":
        distribution, link = "ordered_logistic", "cumulative_logit"
    elif dtype == "count":
        dispersion = float(profile.get("variance_to_mean_ratio") or 1)
        distribution = "negative_binomial" if dispersion > 1.35 else "poisson"
        link = "log"
    elif dtype == "continuous":
        mean = float(profile.get("mean") or 0)
        sd = max(float(profile.get("std") or 0), 1e-6)
        minimum = profile.get("min")
        maximum = profile.get("max")
        extreme = (
            isinstance(minimum, int | float)
            and isinstance(maximum, int | float)
            and (float(maximum) > mean + 5 * sd or float(minimum) < mean - 5 * sd)
        )
        strictly_positive = isinstance(minimum, int | float) and float(minimum) > 0
        if strictly_positive and indicator.get("summary_operator") in {"mean", "last"}:
            distribution, link = "gamma", "log"
        elif extreme:
            distribution, link = "student_t", "identity"
        else:
            distribution, link = "gaussian", "identity"
    else:
        raise ValueError(f"Unsupported DEMO measurement dtype: {dtype}")

    standardized = _indicator_standardized(indicator, distribution)
    profile_summary = (
        f"n={int(profile.get('n_obs') or 0)}, mean={float(profile.get('mean') or 0):.2f}, "
        f"sd={float(profile.get('std') or 0):.2f}"
    )
    return {
        "variable": indicator["name"],
        "distribution": distribution,
        "link": link,
        "standardized": standardized,
        "reasoning": (
            f"{dtype} support with {indicator.get('support_kind')} / "
            f"{indicator.get('summary_operator')} semantics; empirical profile {profile_summary}."
        ),
        "sources": [],
    }


def _reference_indicators(indicators: list[JsonObject]) -> dict[str, str]:
    grouped: dict[str, list[tuple[int, JsonObject]]] = defaultdict(list)
    for index, indicator in enumerate(indicators):
        grouped[str(indicator["construct_name"])].append((index, indicator))
    dtype_tier = {"continuous": 0, "ordinal": 1, "binary": 2, "count": 2, "categorical": 3}
    references: dict[str, str] = {}
    for construct, rows in grouped.items():
        _, reference = min(
            rows,
            key=lambda row: (
                dtype_tier.get(str(row[1].get("measurement_dtype")), 2),
                0 if row[1].get("construct_polarity") == "positive" else 1,
                row[0],
            ),
        )
        references[construct] = str(reference["name"])
    return references


def _parameter(
    name: str,
    role: str,
    constraint: str,
    description: str,
) -> JsonObject:
    return {"name": name, "role": role, "constraint": constraint, "description": description}


def _build_parameters(
    constructs: list[JsonObject],
    edges: list[JsonObject],
    indicators: list[JsonObject],
    likelihoods: list[JsonObject],
) -> list[JsonObject]:
    likelihood_by_name = {str(item["variable"]): item for item in likelihoods}
    indicators_by_construct: dict[str, list[JsonObject]] = defaultdict(list)
    for indicator in indicators:
        indicators_by_construct[str(indicator["construct_name"])].append(indicator)
    references = _reference_indicators(indicators)

    parameters: list[JsonObject] = []
    for indicator in indicators:
        construct = str(indicator["construct_name"])
        likelihood = likelihood_by_name[str(indicator["name"])]
        if len(indicators_by_construct[construct]) > 1 and likelihood["distribution"] in {
            "gaussian",
            "student_t",
        }:
            parameters.append(
                _parameter(
                    f"obs_sd_{indicator['name']}",
                    "measurement_error_sd",
                    "positive",
                    f"Measurement-error SD for {indicator['name']}",
                )
            )

    for indicator in indicators:
        likelihood = likelihood_by_name[str(indicator["name"])]
        family = str(likelihood["distribution"])
        needs_intercept = family in {
            "poisson",
            "negative_binomial",
            "bernoulli",
            "gamma",
            "beta",
        } or (family in {"gaussian", "student_t"} and not likelihood["standardized"])
        if needs_intercept:
            parameters.append(
                _parameter(
                    f"manifest_mean_{indicator['name']}",
                    "observation_intercept",
                    "none",
                    f"Observation intercept for {indicator['name']}",
                )
            )

    active_families = {str(item["distribution"]) for item in likelihoods}
    for family, name, description in (
        ("student_t", "obs_df", "Student-t observation degrees of freedom"),
        ("gamma", "obs_shape", "Gamma observation shape"),
        ("negative_binomial", "obs_r", "Negative-binomial observation dispersion"),
        ("beta", "obs_concentration", "Beta observation concentration"),
    ):
        if family in active_families:
            parameters.append(
                _parameter(
                    name,
                    "observation_hyperparameter_positive",
                    "positive",
                    description,
                )
            )

    for indicator in indicators:
        if likelihood_by_name[str(indicator["name"])]["distribution"] != "ordered_logistic":
            continue
        parameters.append(
            _parameter(
                f"obs_ordered_base_{indicator['name']}",
                "observation_hyperparameter",
                "none",
                f"Ordered-logistic threshold base for {indicator['name']}",
            )
        )
        if len(indicator.get("ordinal_levels") or []) > 2:
            parameters.append(
                _parameter(
                    f"obs_ordered_gaps_{indicator['name']}",
                    "observation_hyperparameter_positive",
                    "positive",
                    f"Ordered-logistic threshold gaps for {indicator['name']}",
                )
            )

    for construct in constructs:
        if construct.get("temporal_status") == "time_varying":
            name = str(construct["name"])
            parameters.append(
                _parameter(
                    f"rho_{name}",
                    "ar_coefficient",
                    "unit_interval",
                    f"Baseline daily persistence absent incoming feedback for {name}",
                )
            )

    for edge in edges:
        cause, effect = str(edge["cause"]), str(edge["effect"])
        lag = "lagged" if edge.get("lagged", True) else "contemporaneous"
        parameters.append(
            _parameter(
                f"beta_{cause}_{effect}",
                "fixed_effect",
                "none",
                f"Effect of {cause} on {effect} ({lag})",
            )
        )

    for construct in constructs:
        if construct.get("temporal_status") == "time_varying":
            name = str(construct["name"])
            parameters.append(
                _parameter(
                    f"sigma_{name}",
                    "residual_sd",
                    "positive",
                    f"Residual/innovation SD for {name}",
                )
            )

    standardized_constructs = {
        str(indicator["construct_name"])
        for indicator in indicators
        if likelihood_by_name[str(indicator["name"])]["standardized"]
    }
    for construct in constructs:
        if construct.get("temporal_status") != "time_invariant":
            continue
        name = str(construct["name"])
        if name in standardized_constructs:
            parameters.append(
                _parameter(
                    f"t0_mean_{name}",
                    "initial_state_mean",
                    "none",
                    f"Initial-state mean for time-invariant construct {name}",
                )
            )
        parameters.append(
            _parameter(
                f"t0_sd_{name}",
                "initial_state_sd",
                "positive",
                f"Initial-state SD for time-invariant construct {name}",
            )
        )

    for indicator in indicators:
        construct = str(indicator["construct_name"])
        if len(indicators_by_construct[construct]) <= 1:
            continue
        if str(indicator["name"]) == references[construct]:
            continue
        parameters.append(
            _parameter(
                f"lambda_{indicator['name']}_{construct}",
                "loading",
                str(indicator["construct_polarity"]),
                f"Factor loading for {indicator['name']} on {construct}",
            )
        )

    names = [str(item["name"]) for item in parameters]
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError(f"Duplicate generated model parameters: {duplicates}")
    return parameters


def _edge_sign(description: str) -> float:
    text = description.lower()
    negative_markers = (
        "reduce",
        "lower",
        "decrease",
        "protect",
        "improv",
        "buffer",
        "mitigat",
        "suppress",
        "stabili",
        "reliev",
        "prevent",
    )
    positive_markers = (
        "increase",
        "raise",
        "worsen",
        "heighten",
        "amplif",
        "exacerbat",
        "trigger",
        "destabili",
    )
    if any(marker in text for marker in negative_markers):
        return -1
    if any(marker in text for marker in positive_markers):
        return 1
    return 1 if _seed_fraction(description) >= 0.32 else -1


def _edge_mean(cause: str, effect: str, description: str) -> float:
    important: dict[tuple[str, str], float] = {
        ("taper_speed_dose_reduction", "serotonergic_change_disequilibrium"): 0.34,
        ("serotonergic_change_disequilibrium", "withdrawal_symptom_burden"): 0.41,
        ("withdrawal_symptom_burden", "internalizing_symptom_burden"): 0.38,
        ("escitalopram_dose_taken", "plasma_escitalopram_exposure"): 0.43,
        ("plasma_escitalopram_exposure", "serotonin_transporter_occupancy"): 0.37,
        ("serotonin_transporter_occupancy", "antidepressant_maintenance_effect"): 0.31,
        ("antidepressant_maintenance_effect", "depressive_anxiety_disorder_activity"): -0.34,
        ("depressive_anxiety_disorder_activity", "internalizing_symptom_burden"): 0.46,
        ("sleep_circadian_disruption", "internalizing_symptom_burden"): 0.29,
        ("perceived_stress_burden", "internalizing_symptom_burden"): 0.31,
        ("rumination_cognitive_reactivity", "internalizing_symptom_burden"): 0.33,
        ("behavioral_activation_reward", "internalizing_symptom_burden"): -0.27,
        ("physical_activity", "internalizing_symptom_burden"): -0.18,
        ("psychotherapy_skills_use", "rumination_cognitive_reactivity"): -0.24,
        ("clinical_monitoring_rescue_care", "acute_safety_risk"): -0.29,
    }
    if (cause, effect) in important:
        return important[(cause, effect)]
    magnitude = 0.06 + 0.12 * _seed_fraction(f"edge:{cause}:{effect}")
    return _round(_edge_sign(description) * magnitude)


def _prior_for(parameter: JsonObject, edge_means: dict[str, float]) -> JsonObject:
    name = str(parameter["name"])
    role = str(parameter["role"])
    constraint = str(parameter["constraint"])
    reference_interval_days: float | None = None

    if role == "ar_coefficient":
        mean = 0.68 + 0.22 * _seed_fraction(name)
        strength = 14.0
        distribution = "Beta"
        params: JsonObject = {
            "alpha": _round(mean * strength),
            "beta": _round((1 - mean) * strength),
        }
        reference_interval_days = 1.0
    elif role == "fixed_effect":
        distribution = "Normal"
        params = {"mu": edge_means[name], "sigma": 0.16}
        reference_interval_days = 1.0
    elif role == "loading":
        sign = -1 if constraint == "negative" else 1
        distribution = "TruncatedNormal"
        params = {
            "mu": float(sign),
            "sigma": 0.3,
            "lower": -3.0 if sign < 0 else 0.05,
            "upper": -0.05 if sign < 0 else 3.0,
        }
    elif role == "observation_hyperparameter":
        distribution = "Normal"
        params = {"mu": 0.0, "sigma": 1.25}
    elif name == "obs_r":
        distribution = "Gamma"
        params = {"concentration": 3.0, "rate": 0.6}
    elif name == "obs_shape":
        distribution = "Gamma"
        params = {"concentration": 5.0, "rate": 1.0}
    elif name == "obs_df":
        distribution = "Gamma"
        params = {"concentration": 4.0, "rate": 0.35}
    elif role in {
        "residual_sd",
        "measurement_error_sd",
        "initial_state_sd",
        "observation_hyperparameter_positive",
    }:
        distribution = "HalfNormal"
        params = {"sigma": 0.65 if role == "observation_hyperparameter_positive" else 0.45}
    else:
        distribution = "Normal"
        params = {"mu": 0.0, "sigma": 1.0}

    proposal: JsonObject = {
        "parameter": name,
        "distribution": distribution,
        "params": params,
        "sources": [],
        "reasoning": (
            "Weakly regularizing deterministic DEMO-fixture prior on the parameter's compiled "
            "scale; suitable for visual review but not scientific evidence."
        ),
        "reference_interval_days": reference_interval_days,
        "density_points": _density_points(distribution, params),
    }
    return proposal


def _draw_likelihood_samples(
    indicator: JsonObject,
    likelihood: JsonObject,
    profile: JsonObject,
) -> list[float]:
    rng = _rng(f"prior-predictive:{indicator['name']}")
    family = str(likelihood["distribution"])
    mean = float(profile.get("mean") or 0)
    sd = max(float(profile.get("std") or 1), 0.1)
    levels = len(indicator.get("ordinal_levels") or [])
    samples: list[float] = []
    for _ in range(PRIOR_SAMPLE_COUNT):
        value = rng.gauss(mean, sd * 1.15)
        if family == "bernoulli":
            probability = min(0.95, max(0.05, mean))
            value = float(rng.random() < probability)
        elif family in {"poisson", "negative_binomial"}:
            value = float(max(0, round(value)))
        elif family == "ordered_logistic":
            upper = max(1, levels - 1)
            value = float(min(upper, max(0, round(value))))
        elif family == "gamma":
            value = max(0.001, value)
        samples.append(_round(value))
    return samples


def _build_statistical_model_spec(
    constructs: list[JsonObject],
    edges: list[JsonObject],
    indicators: list[JsonObject],
    profiles: dict[str, JsonObject],
) -> tuple[JsonObject, dict[str, float]]:
    likelihoods = [
        _likelihood_for(indicator, profiles[str(indicator["name"])]) for indicator in indicators
    ]
    parameters = _build_parameters(constructs, edges, indicators, likelihoods)
    edge_means = {
        f"beta_{edge['cause']}_{edge['effect']}": _edge_mean(
            str(edge["cause"]), str(edge["effect"]), str(edge.get("description") or "")
        )
        for edge in edges
    }
    priors = [_prior_for(parameter, edge_means) for parameter in parameters]
    prior_by_name = {str(prior["parameter"]): prior for prior in priors}
    samples = {
        str(indicator["name"]): _draw_likelihood_samples(
            indicator,
            next(item for item in likelihoods if item["variable"] == indicator["name"]),
            profiles[str(indicator["name"])],
        )
        for indicator in indicators
    }
    diagnostics = []
    for construct in constructs:
        name = str(construct["name"])
        construct_indicators = [
            indicator for indicator in indicators if indicator["construct_name"] == name
        ]
        diagnostics.append(
            {
                "check": "construct_admission",
                "target": name,
                "value": (
                    f"{len(construct_indicators)} channel(s); "
                    f"{sum(len(samples[str(item['name'])]) for item in construct_indicators)} draws"
                ),
                "band": "finite exact prior-predictive draws with support respected",
                "passed": True,
                "note": "Deterministic fixture preview of the construct-level admission battery.",
                "diagnosis": [],
                "mode": "exact_engine_fixture_projection",
            }
        )

    sparse = [
        name
        for name, profile in profiles.items()
        if int(profile.get("n_obs") or 0) < 12 or float(profile.get("zero_fraction") or 0) > 0.97
    ]
    payload: JsonObject = {
        "statistical_model_spec": {
            "likelihoods": likelihoods,
            "parameters": parameters,
            "initialization_policy": "stationary",
            "observation_intercept_policy": "free",
            "equilibrium_forcing": False,
        },
        "authored_priors": prior_by_name,
        "resolved_priors": priors,
        "search_queries": None,
        "validation_warnings": [
            "Artificial downstream DEMO completion: values are for Storybook, not inference.",
            f"Sparse or near-degenerate channels retained for UI coverage: {', '.join(sparse[:8])}.",
        ],
        "prior_predictive_samples": samples,
        "prior_predictive_diagnostics": diagnostics,
    }
    StatisticalModelSpecContract.model_validate(payload)
    return payload, edge_means


def _panel_windows(
    indicator_names: list[str],
) -> tuple[dict[str, list[float | None]], int, str]:
    panel = pl.read_parquet(PANEL_PATH).sort(["indicator", "anchor_time"])
    unique_times = panel.select(pl.col("anchor_time").n_unique()).item()
    final_time = panel.select(pl.col("anchor_time").max()).item()
    if final_time is None:
        raise ValueError("DEMO panel has no anchor times")
    values_by_indicator: dict[str, list[float | None]] = {}
    for name in indicator_names:
        values = panel.filter(pl.col("indicator") == name).get_column("value").to_list()
        non_null_indices = [index for index, value in enumerate(values) if value is not None]
        end = (non_null_indices[-1] + 1) if non_null_indices else len(values)
        start = max(0, end - PPC_POINTS)
        window = values[start:end]
        if len(window) < PPC_POINTS:
            window = [None] * (PPC_POINTS - len(window)) + window
        values_by_indicator[name] = [None if value is None else _round(value) for value in window]
    return values_by_indicator, int(unique_times), final_time.isoformat()


def _bounded(value: float, likelihood: JsonObject, indicator: JsonObject) -> float:
    family = str(likelihood["distribution"])
    if family == "bernoulli":
        return min(1.0, max(0.0, value))
    if family in {"poisson", "negative_binomial", "gamma"}:
        return max(0.0, value)
    if family == "ordered_logistic":
        upper = max(1, len(indicator.get("ordinal_levels") or []) - 1)
        return min(float(upper), max(0.0, value))
    return value


def _ppc_for_indicator(
    indicator: JsonObject,
    likelihood: JsonObject,
    profile: JsonObject,
    observed: list[float | None],
) -> tuple[JsonObject, list[JsonObject], list[JsonObject]]:
    name = str(indicator["name"])
    rng = _rng(f"ppc:{name}")
    mean = float(profile.get("mean") or 0)
    sd = max(float(profile.get("std") or 0), 0.08)
    family = str(likelihood["distribution"])
    phase = 2 * math.pi * _seed_fraction(f"phase:{name}")
    median: list[float] = []
    for index, value in enumerate(observed):
        trend = mean + 0.12 * sd * math.sin(index / 8 + phase)
        if value is not None:
            trend = 0.68 * float(value) + 0.32 * trend
        median.append(_round(_bounded(trend, likelihood, indicator)))

    width = max(0.12, 0.72 * sd)
    if family == "bernoulli":
        width = 0.48
    q025 = [_round(_bounded(value - 1.55 * width, likelihood, indicator)) for value in median]
    q25 = [_round(_bounded(value - 0.48 * width, likelihood, indicator)) for value in median]
    q75 = [_round(_bounded(value + 0.48 * width, likelihood, indicator)) for value in median]
    q975 = [_round(_bounded(value + 1.55 * width, likelihood, indicator)) for value in median]
    spaghetti: list[list[float]] = []
    for _ in range(4):
        row = []
        carry = 0.0
        for value in median:
            carry = 0.55 * carry + rng.gauss(0, width * 0.35)
            draw = _bounded(value + carry, likelihood, indicator)
            if family in {"bernoulli", "poisson", "negative_binomial", "ordered_logistic"}:
                draw = round(draw)
            row.append(_round(draw))
        spaghetti.append(row)

    overlay = {
        "variable": name,
        "observed": observed,
        "q025": q025,
        "q25": q25,
        "median": median,
        "q75": q75,
        "q975": q975,
        "spaghetti_draws": spaghetti,
    }

    n_obs = int(profile.get("n_obs") or 0)
    zero_fraction = float(profile.get("zero_fraction") or 0)
    calibration = 0.91 + 0.06 * _seed_fraction(f"coverage:{name}")
    residual_rho = -0.18 + 0.36 * _seed_fraction(f"rho:{name}")
    variance_ratio = 0.78 + 0.45 * _seed_fraction(f"variance:{name}")
    calibration_passed = n_obs >= 8 and zero_fraction < 0.98
    warnings = [
        {
            "variable": name,
            "check_type": "calibration",
            "message": (
                f"95% interval coverage {calibration:.1%}."
                if calibration_passed
                else "Calibration is weakly assessed because the observed channel is sparse or degenerate."
            ),
            "value": _round(calibration),
            "passed": calibration_passed,
        },
        {
            "variable": name,
            "check_type": "autocorrelation",
            "message": f"Lag-1 residual autocorrelation {residual_rho:+.2f}.",
            "value": _round(residual_rho),
            "passed": abs(residual_rho) <= 0.3,
        },
        {
            "variable": name,
            "check_type": "variance",
            "message": f"Predictive-to-observed SD ratio {variance_ratio:.2f}.",
            "value": _round(variance_ratio),
            "passed": 1 / 3 <= variance_ratio <= 3,
        },
    ]

    observed_values = [float(value) for value in observed if value is not None]
    if not observed_values:
        observed_values = [mean]
    statistics_by_name = {
        "mean": statistics.fmean(observed_values),
        "sd": statistics.pstdev(observed_values) if len(observed_values) > 1 else 0.0,
        "min": min(observed_values),
        "max": max(observed_values),
    }
    test_stats = []
    for stat_name, observed_value in statistics_by_name.items():
        rep_sd = max(sd * (0.08 if stat_name in {"mean", "sd"} else 0.18), 0.03)
        rep_values = [_round(rng.gauss(observed_value, rep_sd)) for _ in range(64)]
        test_stats.append(
            {
                "variable": name,
                "stat_name": stat_name,
                "observed_value": _round(observed_value),
                "rep_values": rep_values,
            }
        )
    return overlay, warnings, test_stats


def _prior_center(prior: JsonObject) -> tuple[float, float]:
    distribution = str(prior["distribution"])
    params = prior["params"]
    if distribution == "Normal":
        return float(params["mu"]), float(params["sigma"])
    if distribution == "HalfNormal":
        sigma = float(params["sigma"])
        return sigma * math.sqrt(2 / math.pi), sigma * 0.45
    if distribution == "Beta":
        alpha, beta = float(params["alpha"]), float(params["beta"])
        mean = alpha / (alpha + beta)
        variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
        return mean, math.sqrt(variance)
    if distribution == "Gamma":
        concentration, rate = float(params["concentration"]), float(params["rate"])
        return concentration / rate, math.sqrt(concentration) / rate
    if distribution == "TruncatedNormal":
        return float(params["mu"]), float(params["sigma"])
    raise ValueError(distribution)


def _posterior_center(parameter: JsonObject, prior: JsonObject) -> tuple[float, float]:
    mean, prior_sd = _prior_center(prior)
    role = str(parameter["role"])
    if role == "fixed_effect":
        mean *= 0.92 + 0.12 * _seed_fraction(f"posterior:{parameter['name']}")
    elif role == "ar_coefficient":
        mean = min(0.97, mean + 0.025 * (_seed_fraction(str(parameter["name"])) - 0.5))
    elif role == "loading":
        mean *= 0.96 + 0.08 * _seed_fraction(str(parameter["name"]))
    sd = max(0.018, prior_sd * (0.28 + 0.14 * _seed_fraction(f"sd:{parameter['name']}")))
    return _round(mean), _round(sd)


def _marginal(parameter: JsonObject, prior: JsonObject) -> JsonObject:
    mean, sd = _posterior_center(parameter, prior)
    constraint = str(parameter["constraint"])
    lower = mean - 3.4 * sd
    upper = mean + 3.4 * sd
    if constraint in {"positive", "unit_interval"}:
        lower = max(0.001, lower)
    if constraint == "unit_interval":
        upper = min(0.999, upper)
    if constraint == "negative":
        upper = min(-0.001, upper)
    xs = _linspace(lower, upper, MARGINAL_POINTS)
    density = [_normal_pdf(value, mean, sd) for value in xs]
    return {
        "parameter": parameter["name"],
        "x_values": [_round(value) for value in xs],
        "density": [_round(value) for value in density],
        "mean": mean,
        "sd": sd,
        "hdi_3": _round(max(lower, mean - 1.88 * sd)),
        "hdi_97": _round(min(upper, mean + 1.88 * sd)),
    }


def _build_posterior(
    indicators: list[JsonObject],
    profiles: dict[str, JsonObject],
    statistical_model_spec: JsonObject,
    panel_windows: dict[str, list[float | None]],
    n_timesteps: int,
) -> JsonObject:
    likelihoods = statistical_model_spec["statistical_model_spec"]["likelihoods"]
    likelihood_by_name = {str(item["variable"]): item for item in likelihoods}
    overlays: list[JsonObject] = []
    warnings: list[JsonObject] = []
    test_stats: list[JsonObject] = []
    for indicator in indicators:
        name = str(indicator["name"])
        overlay, indicator_warnings, indicator_stats = _ppc_for_indicator(
            indicator,
            likelihood_by_name[name],
            profiles[name],
            panel_windows[name],
        )
        overlays.append(overlay)
        warnings.extend(indicator_warnings)
        test_stats.extend(indicator_stats)

    parameters = statistical_model_spec["statistical_model_spec"]["parameters"]
    priors = statistical_model_spec["authored_priors"]
    marginals = [_marginal(parameter, priors[str(parameter["name"])]) for parameter in parameters]
    marginal_by_name = {str(item["parameter"]): item for item in marginals}
    diagnostics = []
    for parameter in parameters:
        name = str(parameter["name"])
        r_hat = 1.0 + 0.009 * _seed_fraction(f"rhat:{name}")
        ess_bulk = 850 + 2300 * _seed_fraction(f"ess:{name}")
        diagnostics.append(
            {
                "parameter": name,
                "r_hat": _round(r_hat, 4),
                "ess_bulk": _round(ess_bulk, 1),
                "ess_tail": _round(ess_bulk * (0.72 + 0.15 * _seed_fraction(name)), 1),
                "mcse_mean": _round(marginal_by_name[name]["sd"] / math.sqrt(ess_bulk), 6),
            }
        )

    focus_names = [
        name
        for name in (
            "rho_internalizing_symptom_burden",
            "sigma_internalizing_symptom_burden",
            "beta_escitalopram_dose_taken_internalizing_symptom_burden",
            "beta_taper_speed_dose_reduction_escitalopram_dose_taken",
            "beta_adherence_to_regimen_escitalopram_dose_taken",
            "beta_sleep_circadian_disruption_internalizing_symptom_burden",
            "beta_perceived_stress_burden_internalizing_symptom_burden",
            "beta_physical_activity_internalizing_symptom_burden",
            "beta_external_stressful_events_perceived_stress_burden",
        )
        if name in marginal_by_name
    ]
    trace_data = []
    rank_histograms = []
    for name in focus_names:
        marginal = marginal_by_name[name]
        chains = []
        rank_chains = []
        for chain in range(4):
            rng = _rng(f"trace:{name}:{chain}")
            values: list[float] = []
            carry = 0.0
            for _ in range(72):
                carry = 0.45 * carry + rng.gauss(0, float(marginal["sd"]))
                values.append(_round(float(marginal["mean"]) + carry))
            chains.append({"chain": chain, "values": values})
            rank_rng = _rng(f"rank:{name}:{chain}")
            rank_chains.append(
                {
                    "chain": chain,
                    "counts": [max(1, 18 + rank_rng.randint(-5, 5)) for _ in range(10)],
                }
            )
        trace_data.append({"parameter": name, "chains": chains})
        rank_histograms.append(
            {"parameter": name, "n_bins": 10, "expected_per_bin": 18.0, "chains": rank_chains}
        )

    pair_names = list(zip(focus_names[::2], focus_names[1::2], strict=False))[:4]
    pairs = []
    for left, right in pair_names:
        left_marginal, right_marginal = marginal_by_name[left], marginal_by_name[right]
        rng = _rng(f"pair:{left}:{right}")
        xs, ys = [], []
        correlation = -0.35 + 0.7 * _seed_fraction(f"pair-correlation:{left}:{right}")
        for _ in range(180):
            z1 = rng.gauss(0, 1)
            z2 = correlation * z1 + math.sqrt(1 - correlation**2) * rng.gauss(0, 1)
            xs.append(_round(float(left_marginal["mean"]) + float(left_marginal["sd"]) * z1))
            ys.append(_round(float(right_marginal["mean"]) + float(right_marginal["sd"]) * z2))
        pairs.append(
            {
                "param_x": left,
                "param_y": right,
                "x_values": xs,
                "y_values": ys,
                "divergent": [False] * len(xs),
            }
        )

    pareto_rng = _rng("loo-pareto")
    loo_points = min(180, n_timesteps)
    payload: JsonObject = {
        "ppc": {
            "per_variable_warnings": warnings,
            "checked": True,
            "n_subsample": 128,
            "overlays": overlays,
            "test_stats": test_stats,
        },
        "inference_metadata": {
            "method": "marginal_particle_gibbs",
            "n_samples": 3200,
            "duration_seconds": 1842.7,
        },
        "mcmc_diagnostics": {
            "per_parameter": diagnostics,
            "num_divergences": 0,
            "divergence_rate": 0.0,
            "tree_depth_mean": 0.0,
            "tree_depth_max": 0,
            "accept_prob_mean": 0.84,
            "num_chains": 4,
            "num_samples": 800,
            "trace_data": trace_data,
            "rank_histograms": rank_histograms,
            "energy": None,
        },
        "smc_diagnostics": {
            "beta_schedule": [0.0, 0.08, 0.19, 0.34, 0.52, 0.71, 0.86, 1.0],
            "ess_history": [512.0, 474.0, 438.0, 401.0, 365.0, 337.0, 318.0, 301.0],
            "accept_rates": [0.91, 0.88, 0.86, 0.83, 0.81, 0.78, 0.76, 0.74],
            "n_levels": 8,
            "n_particles": 512,
        },
        "loo_diagnostics": {
            "elpd_loo": -4287.4,
            "p_loo": 186.3,
            "se": 74.8,
            "n_data_points": n_timesteps,
            "observation_unit": "timestep",
            "pareto_k": [_round(0.08 + 0.62 * pareto_rng.random()) for _ in range(loo_points)],
            "n_bad_k": 0,
            "loo_pit": [_round(0.02 + 0.96 * pareto_rng.random()) for _ in range(loo_points)],
        },
        "posterior_marginals": marginals,
        "posterior_pairs": pairs,
    }
    return payload


TREATMENT_EFFECTS: dict[str, float] = {
    "depressive_anxiety_disorder_activity": 0.48,
    "escitalopram_dose_taken": -0.18,
    "taper_speed_dose_reduction": 0.36,
    "adherence_to_regimen": -0.16,
    "plasma_escitalopram_exposure": -0.17,
    "serotonin_transporter_occupancy": -0.2,
    "antidepressant_maintenance_effect": -0.4,
    "neuroadaptation_dependence_state": 0.13,
    "serotonergic_change_disequilibrium": 0.42,
    "withdrawal_symptom_burden": 0.54,
    "escitalopram_side_effect_burden": 0.24,
    "expectancy_nocebo_placebo": 0.16,
    "sleep_circadian_disruption": 0.38,
    "perceived_stress_burden": 0.34,
    "external_stressful_events": 0.28,
    "rumination_cognitive_reactivity": 0.36,
    "behavioral_activation_reward": -0.29,
    "physical_activity": -0.21,
    "social_support_conflict": -0.16,
    "psychotherapy_skills_use": -0.25,
    "substance_use_intoxication_withdrawal": 0.23,
    "physical_illness_pain_inflammation": 0.24,
    "hormonal_state_changes": 0.11,
    "role_functioning_demands": 0.2,
    "acute_safety_risk": 0.45,
    "clinical_monitoring_rescue_care": -0.19,
    "other_psychotropic_or_somatic_treatments": -0.08,
    "patient_taper_preference_beliefs": 0.08,
    "clinician_taper_guidance": -0.18,
    "counterfactual_taper_regime": 0.31,
    "baseline_depression_anxiety_severity_chronicity": 0.41,
    "prior_episode_recurrence_history": 0.31,
    "comorbid_psychiatric_vulnerability": 0.3,
    "relapse_vulnerability": 0.43,
    "natural_recovery_propensity": -0.29,
    "pre_taper_remission_stability": -0.31,
    "pre_taper_residual_symptom_burden": 0.36,
    "past_escitalopram_response_tolerability": -0.18,
    "duration_current_escitalopram_use": 0.1,
    "baseline_maintenance_dose_level": -0.1,
    "stable_ssri_pharmacodynamic_responsiveness": -0.21,
    "stable_withdrawal_susceptibility": 0.35,
    "prior_discontinuation_withdrawal_history": 0.29,
    "genetic_family_liability": 0.25,
    "early_life_adversity": 0.28,
    "temperament_neuroticism_behavioral_inhibition": 0.32,
    "cyp2c19_pharmacokinetic_capacity": 0.05,
    "age_life_stage": 0.04,
    "sex_reproductive_context": 0.03,
    "stable_socioeconomic_resources": -0.22,
    "baseline_physical_health_burden": 0.25,
    "healthcare_access": -0.18,
    "medication_formulation_constraints": 0.14,
    "stable_medication_attitudes": 0.05,
}

ARTIFICIAL_PROFILE_OVERRIDES: dict[str, JsonObject] = {
    "hormonal_change_mention": {
        "measurement_dtype": "binary",
        "n_obs": 0,
        "mean": 0.0,
        "std": 0.0,
        "min": 0.0,
        "max": 0.0,
        "q25": 0.0,
        "q50": 0.0,
        "q75": 0.0,
        "variance": 0.0,
        "time_coverage_ratio": 0.0,
        "max_gap_ratio": 1.0,
        "dtype_violations": 0,
        "duplicate_pct": 1.0,
        "arithmetic_sequence_detected": False,
        "n_unparseable_timestamps": 0,
        "zero_fraction": 1.0,
        "is_nonnegative": True,
        "is_unit_interval": True,
        "looks_integer_valued": True,
        "variance_to_mean_ratio": 0.0,
    }
}


def _manifest_effects(effect: float) -> dict[str, float]:
    return {
        "phq9_screening_score": _round(effect * 4.2),
        "gad7_screening_score": _round(effect * 3.6),
        "state_of_mind_valence": _round(-effect * 0.65),
        "journal_internalizing_symptom_severity": _round(effect * 0.8),
    }


def _treatment_result_mean(item: JsonObject) -> float:
    draws = item["posterior_draws"]
    if not isinstance(draws, list):
        raise TypeError("Generated treatment posterior_draws must be a list")
    return statistics.fmean(float(value) for value in draws)


def _build_baseline_report(identifiable_treatments: list[str]) -> JsonObject:
    missing = sorted(set(identifiable_treatments) - set(TREATMENT_EFFECTS))
    if missing:
        raise ValueError(f"Missing fixture treatment-effect direction for {missing}")
    results = []
    for treatment in identifiable_treatments:
        mean = TREATMENT_EFFECTS[treatment] + 0.012 * (_seed_fraction(treatment) - 0.5)
        rng = _rng(f"treatment:{treatment}")
        sd = 0.075 + 0.045 * abs(mean)
        draws = [_round(rng.gauss(mean, sd)) for _ in range(POSTERIOR_DRAW_COUNT)]
        peak_day = 18 + round(22 * _seed_fraction(f"peak:{treatment}"))
        results.append(
            {
                "treatment": treatment,
                "posterior_draws": draws,
                "temporal": {
                    "effect_1d": _round(mean * 0.14),
                    "effect_7d": _round(mean * 0.52),
                    "effect_30d": _round(mean * 0.94),
                    "peak_effect": _round(mean * 1.05),
                    "time_to_peak_days": float(peak_day),
                },
                "manifest_effects": _manifest_effects(mean),
            }
        )

    results.sort(key=lambda item: abs(_treatment_result_mean(item)), reverse=True)
    return {
        "intervention_results": results,
        "saved_scenarios": [
            {
                "label": "Rapid taper",
                "query": "What happens if taper speed is raised sharply from the baseline state?",
                "summary": (
                    "The executable preview follows the measured dose-maintenance path; the "
                    "unmeasured withdrawal branch remains visible but is not simulated."
                ),
            },
            {
                "label": "Gradual taper",
                "query": "Compare a deliberately slow taper with the current plan.",
                "summary": (
                    "A slower reduction spreads the modeled dose change. Withdrawal remains "
                    "outside the executable state, so this is not a complete taper-effect claim."
                ),
            },
            {
                "label": "Adherence support",
                "query": "What if regimen adherence is stabilized at a high level?",
                "summary": "Stable adherence reduces modeled dose volatility in the executable projection.",
            },
            {
                "label": "Sleep stabilization",
                "query": "What if sleep and circadian disruption are reduced for the next month?",
                "summary": "Improved sleep reduces the retained direct symptom path.",
            },
            {
                "label": "Stress pulse",
                "query": "From the latest fitted state, what if a major external stressor occurs in week three?",
                "summary": "A temporary stress pulse raises perceived stress and disrupts sleep before fading.",
            },
        ],
        "final_summary": (
            "Artificial DEMO completion for visual review. The executable preview contains measured "
            "taper, dose, adherence, sleep, activity, stress, and symptom states. Neuroadaptation, "
            "withdrawal, natural recovery, and taper-decision context remain visible in the scientific "
            "DAG but are excluded from fitted dynamics. The deterministic numbers are UI fixture data, "
            "not patient claims or a complete taper-effect estimate."
        ),
    }


def _descendants(edges: list[JsonObject], start: str) -> dict[str, int]:
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        successors[str(edge["cause"])].append(str(edge["effect"]))
    distances = {start: 0}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for successor in successors[node]:
            if successor in distances:
                continue
            distances[successor] = distances[node] + 1
            queue.append(successor)
    return distances


def _baseline_level(name: str) -> float:
    return 0.32 + 0.36 * _seed_fraction(f"baseline:{name}")


def _reference_trajectories(constructs: list[JsonObject]) -> dict[str, list[float]]:
    trajectories: dict[str, list[float]] = {}
    for construct in constructs:
        name = str(construct["name"])
        level = _baseline_level(name)
        phase = 2 * math.pi * _seed_fraction(f"reference-phase:{name}")
        varying = construct.get("temporal_status") == "time_varying"
        amplitude = 0.018 + 0.025 * _seed_fraction(f"reference-amplitude:{name}")
        trajectories[name] = [
            _round(level + (amplitude * math.sin(day / 8.5 + phase) if varying else 0))
            for day in range(HORIZON_DAYS + 1)
        ]
    return trajectories


def _clamp_delta(clamp: JsonObject, reference: list[float], day: int) -> float:
    start = float(clamp.get("from_day") or 0)
    end = clamp.get("to_day")
    if day < start or (end is not None and day >= float(end)):
        return 0.0
    mode = str(clamp["mode"])
    if mode == "set":
        return float(clamp["value"]) - reference[day]
    if mode == "shift":
        return float(clamp["amount"])
    if mode == "ramp":
        ramp_end = float(end if end is not None else HORIZON_DAYS)
        progress = min(1.0, max(0.0, (day - start) / max(1.0, ramp_end - start)))
        value = float(clamp["value_start"]) + progress * (
            float(clamp["value_end"]) - float(clamp["value_start"])
        )
        return value - reference[day]
    raise ValueError(f"Unsupported fixture clamp mode: {mode}")


def _simulate_preview(
    *,
    scenario_id: str,
    constructs: list[JsonObject],
    edges: list[JsonObject],
    reference: dict[str, list[float]],
    clamps: list[JsonObject],
    target_effect: float,
    abducted_time_index: int,
    abducted_time: str,
    abducted: bool = False,
) -> JsonObject:
    construct_by_name = {str(construct["name"]): construct for construct in constructs}
    effects = {str(construct["name"]): [0.0] * (HORIZON_DAYS + 1) for construct in constructs}
    clamp_by_node = {str(clamp["variable"]): clamp for clamp in clamps}

    for clamp in clamps:
        variable = str(clamp["variable"])
        direct = effects[variable]
        for day in range(HORIZON_DAYS + 1):
            direct[day] = _clamp_delta(clamp, reference[variable], day)

        distances = _descendants(edges, variable)
        direct_size = max(abs(value) for value in direct) or 1.0
        for node, distance in distances.items():
            if node == variable or construct_by_name[node].get("temporal_status") != "time_varying":
                continue
            direction = 1 if TREATMENT_EFFECTS.get(node, 0.05) * target_effect >= 0 else -1
            attenuation = 0.34 / max(1, distance)
            onset = int(float(clamp.get("from_day") or 0)) + 2 * distance
            tau = 7 + 3 * distance
            for day in range(HORIZON_DAYS + 1):
                if day < onset:
                    continue
                ramp = 1 - math.exp(-(day - onset + 1) / tau)
                effects[node][day] += direction * attenuation * direct_size * ramp

    outcome = "internalizing_symptom_burden"
    onset = min((int(float(clamp.get("from_day") or 0)) for clamp in clamps), default=0)
    for day in range(HORIZON_DAYS + 1):
        ramp = 0 if day < onset else 1 - math.exp(-(day - onset + 1) / 14)
        effects[outcome][day] = target_effect * ramp

    action = {
        node: [_round(value + effects[node][day]) for day, value in enumerate(series)]
        for node, series in reference.items()
    }
    for variable, clamp in clamp_by_node.items():
        for day in range(HORIZON_DAYS + 1):
            delta = _clamp_delta(clamp, reference[variable], day)
            if delta != 0:
                action[variable][day] = _round(reference[variable][day] + delta)

    trajectory = [
        {"day": day, "effect": _round(effects[outcome][day])} for day in range(HORIZON_DAYS + 1)
    ]
    final = trajectory[-1]["effect"]
    interval_width = 0.075 + 0.22 * abs(final)
    probability_positive = 1 / (1 + math.exp(-final / 0.065)) if final else 0.5
    start_state = {name: _round(series[0]) for name, series in reference.items()}
    return {
        "start": {
            "kind": "abducted" if abducted else "baseline",
            "time_index": abducted_time_index if abducted else None,
            "time": abducted_time if abducted else None,
            "state_source": "fitted_latent_paths" if abducted else "baseline_steady_state",
        },
        "clamps": clamps,
        "outcome": outcome,
        "estimand": "trajectory",
        "summary": {
            "mean": _round(final),
            "median": _round(final * 0.98),
            "lower_95": _round(final - interval_width),
            "upper_95": _round(final + interval_width),
            "prob_positive": _round(probability_positive),
        },
        "effect_trajectory": trajectory,
        "visualization": {
            "reference_node_trajectories": reference,
            "action_node_trajectories": action,
            "node_effect_trajectories": {
                node: [_round(value) for value in values] for node, values in effects.items()
            },
            "start_state": start_state,
        },
        "manifest_effects": _manifest_effects(final),
        "reference_mean": _round(reference[outcome][0]),
        "warnings": [
            f"Artificial Storybook simulation {scenario_id}; not a fitted patient result."
        ],
    }


def _trace_message(
    role: str,
    content: str,
    *,
    reasoning: str | None = None,
    tool_calls: list[JsonObject] | None = None,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
    tool_result: str | None = None,
) -> JsonObject:
    return {
        "role": role,
        "content": content,
        "reasoning": reasoning,
        "tool_calls": tool_calls,
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_result": tool_result,
        "tool_is_error": False,
    }


def _simulate_tool_turn(
    scenario_id: str,
    query: str,
    blurb: str,
    result: JsonObject,
) -> list[JsonObject]:
    tool_input = {
        "start": {
            "kind": result["start"]["kind"],
            **(
                {"time_index": result["start"]["time_index"]}
                if result["start"]["kind"] == "abducted"
                else {}
            ),
        },
        "clamps": result["clamps"],
        "outcome": result["outcome"],
        "query": {
            "estimand": "trajectory",
            "horizon_days": HORIZON_DAYS,
            "projection": "both",
        },
    }
    return [
        _trace_message("user", query),
        _trace_message(
            "assistant",
            blurb,
            reasoning=(
                "Use the materialized causal design and fitted-world preview; keep the full theory "
                "graph visible while the simulation layer supplies only executable dynamics."
            ),
            tool_calls=[
                {
                    "id": scenario_id,
                    "type": "function",
                    "function": {
                        "name": "simulate",
                        "arguments": json.dumps(tool_input, separators=(",", ":")),
                    },
                }
            ],
        ),
        _trace_message(
            "tool",
            "",
            tool_call_id=scenario_id,
            tool_name="simulate",
            tool_result=json.dumps(result, separators=(",", ":")),
        ),
    ]


def _build_baseline_trace(
    constructs: list[JsonObject],
    edges: list[JsonObject],
    final_time_index: int,
    final_time: str,
) -> JsonObject:
    reference = _reference_trajectories(constructs)
    definitions = [
        (
            "sim-1",
            "Compare a deliberately slow taper with the current plan.",
            "**Gradual taper.** Lowering taper speed spreads the measured dose change. The gray withdrawal branch is not executable, so the preview is intentionally incomplete.",
            [
                {
                    "variable": "taper_speed_dose_reduction",
                    "mode": "set",
                    "value": 0.24,
                    "from_day": 0,
                }
            ],
            -0.18,
            False,
        ),
        (
            "sim-2",
            "What if regimen adherence is stabilized at a high level?",
            "**Adherence support.** Stable adherence reduces modeled dose volatility; unsupported neuroadaptation and withdrawal states remain unchanged.",
            [{"variable": "adherence_to_regimen", "mode": "set", "value": 0.94, "from_day": 0}],
            -0.15,
            False,
        ),
        (
            "sim-3",
            "What if sleep and circadian disruption are reduced for the next month?",
            "**Sleep stabilization.** Reducing disruption lowers the retained direct symptom path over the following month.",
            [
                {
                    "variable": "sleep_circadian_disruption",
                    "mode": "shift",
                    "amount": -0.34,
                    "from_day": 0,
                    "to_day": 30,
                }
            ],
            -0.27,
            False,
        ),
        (
            "sim-4",
            "From the latest fitted state, what if perceived stress spikes in week three?",
            "**Abducted stress pulse.** A temporary rise in perceived stress disrupts sleep and leaves a smaller residual symptom effect after the pulse ends.",
            [
                {
                    "variable": "perceived_stress_burden",
                    "mode": "shift",
                    "amount": 0.72,
                    "from_day": 14,
                    "to_day": 22,
                }
            ],
            0.25,
            True,
        ),
        (
            "sim-5",
            "What happens if taper speed is raised sharply from the baseline state?",
            "**Rapid taper.** The executable preview follows only the measured taper-to-dose-to-symptom path. It does not materialize the gray withdrawal mechanism.",
            [
                {
                    "variable": "taper_speed_dose_reduction",
                    "mode": "set",
                    "value": 0.9,
                    "from_day": 0,
                }
            ],
            0.34,
            False,
        ),
    ]
    messages = [
        _trace_message(
            "system",
            "Artificial DEMO analysis trace for Storybook. Values exercise only the current production artifact and simulation contracts; they are not scientific results.",
        )
    ]
    for scenario_id, query, blurb, clamps, target_effect, abducted in definitions:
        result = _simulate_preview(
            scenario_id=scenario_id,
            constructs=constructs,
            edges=edges,
            reference=reference,
            clamps=clamps,
            target_effect=target_effect,
            abducted_time_index=final_time_index,
            abducted_time=final_time,
            abducted=abducted,
        )
        messages.extend(_simulate_tool_turn(scenario_id, query, blurb, result))

    return {
        "model": "fixture/nof1-causal-lab",
        "total_time_seconds": 12.8,
        "usage": {"input_tokens": 6840, "output_tokens": 2140, "reasoning_tokens": 760},
        "messages": messages,
    }


def _build_model_spec_trace(
    constructs: list[JsonObject],
    indicators: list[JsonObject],
    statistical_model_spec: JsonObject,
) -> JsonObject:
    parameters = statistical_model_spec["statistical_model_spec"]["parameters"]
    families: dict[str, int] = defaultdict(int)
    for likelihood in statistical_model_spec["statistical_model_spec"]["likelihoods"]:
        families[str(likelihood["distribution"])] += 1
    family_summary = ", ".join(f"{family}={count}" for family, count in sorted(families.items()))
    return {
        "model": "fixture/nof1-causal-lab",
        "total_time_seconds": 18.6,
        "usage": {"input_tokens": 11240, "output_tokens": 3260, "reasoning_tokens": 1480},
        "messages": [
            _trace_message(
                "system",
                "Artificial DEMO model-spec trace for Storybook. It mirrors the compiler-owned decision surfaces without claiming a completed fit.",
            ),
            _trace_message(
                "user",
                f"Specify the validated daily model with {len(constructs)} constructs and {len(indicators)} retained indicators.",
            ),
            _trace_message(
                "assistant",
                (
                    f"Materialized one likelihood per indicator ({family_summary}) and {len(parameters)} "
                    "compiler-shaped prior surfaces. The model uses stationary initialization, free "
                    "eligible observation intercepts, and no equilibrium forcing. Sparse safety and "
                    "event channels remain visible as warnings rather than being silently removed."
                ),
                reasoning=(
                    "Resolve support-compatible likelihoods from declared dtypes and empirical profiles; "
                    "then enumerate persistence, causal-edge, diffusion, measurement, threshold, loading, "
                    "and static-state parameters in the same semantic namespaces used by compilation."
                ),
            ),
        ],
    }


def _compact_demo_latent(full_latent: JsonObject) -> JsonObject:
    """Select a readable scientific DAG while preserving unsupported mechanisms."""
    construct_lookup = {
        str(construct["name"]): dict(construct) for construct in full_latent["constructs"]
    }
    missing_constructs = sorted(set(DEMO_CONSTRUCT_ORDER) - set(construct_lookup))
    if missing_constructs:
        raise ValueError(f"Stored DEMO theory is missing compact constructs: {missing_constructs}")

    constructs = [construct_lookup[name] for name in DEMO_CONSTRUCT_ORDER]
    for construct in constructs:
        role = DEMO_CONSTRUCT_ROLE_OVERRIDES.get(str(construct["name"]))
        if role is not None:
            construct["role"] = role

    edge_lookup = {
        (str(edge["cause"]), str(edge["effect"])): dict(edge) for edge in full_latent["edges"]
    }
    edges: list[JsonObject] = []
    for endpoints in DEMO_EDGE_ORDER:
        edge = DEMO_REDUCED_EDGES.get(endpoints) or edge_lookup.get(endpoints)
        if edge is None:
            raise ValueError(f"Stored DEMO theory is missing compact edge: {endpoints}")
        edges.append(dict(edge))
    return {"constructs": constructs, "edges": edges}


def _compact_demo_measurement(full_measurement: JsonObject) -> JsonObject:
    """Keep high-value, nonredundant indicators from the existing DEMO panel."""
    indicator_lookup = {
        str(indicator["name"]): dict(indicator) for indicator in full_measurement["indicators"]
    }
    missing_indicators = sorted(set(DEMO_INDICATOR_ORDER) - set(indicator_lookup))
    if missing_indicators:
        raise ValueError(
            f"Stored DEMO measurement structure is missing compact indicators: {missing_indicators}"
        )
    return {
        "indicators": [indicator_lookup[name] for name in DEMO_INDICATOR_ORDER],
        "model_clock": full_measurement["model_clock"],
    }


def _compact_demo_validation(
    full_validation: JsonObject,
    indicator_names: tuple[str, ...],
) -> JsonObject:
    """Project empirical audits to the indicators selected by the compact design."""
    audits = full_validation["indicators"]
    missing = sorted(set(indicator_names) - set(audits))
    if missing:
        raise ValueError(f"Stored DEMO validation is missing compact indicator audits: {missing}")
    payload = {
        "is_valid": True,
        "indicators": {name: audits[name] for name in indicator_names},
        # Cross-indicator warnings in the stored report concern proxy pairs that the
        # compact measurement model deliberately de-duplicates.
        "dataset_issues": [],
    }
    return ValidationReportContract.model_validate(payload).model_dump(mode="json", by_alias=True)


def _canonical_identification_status(identification: JsonObject) -> JsonObject:
    """Sort production identification collections for byte-stable fixture JSON."""

    identifiable: JsonObject = {}
    for treatment in sorted(identification.get("identifiable_treatments", {})):
        status = dict(identification["identifiable_treatments"][treatment])
        status["marginalized_confounders"] = sorted(status.get("marginalized_confounders", []))
        status["instruments"] = sorted(status.get("instruments", []))
        identifiable[treatment] = status

    non_identifiable: JsonObject = {}
    for treatment in sorted(identification.get("non_identifiable_treatments", {})):
        status = dict(identification["non_identifiable_treatments"][treatment])
        status["confounders"] = sorted(status.get("confounders", []))
        non_identifiable[treatment] = status

    return {
        "identifiable_treatments": identifiable,
        "non_identifiable_treatments": non_identifiable,
    }


def _load_sources() -> FixtureSources:
    latent_payload = _read_json(STORE_ROOT / "latent_structure/v1/latent-structure.json")
    measurement_payload = _read_json(
        STORE_ROOT / "measurement_structure/v1/measurement_structure.json"
    )
    stored_validation = _read_json(STORE_ROOT / "validation_report/v1/validation_report.json")
    ValidationReportContract.model_validate(stored_validation)

    latent = _compact_demo_latent(latent_payload["latent_structure"])
    measurement = _compact_demo_measurement(measurement_payload["measurement_structure"])
    known_inputs = [dict(item) for item in DEMO_KNOWN_INPUTS]
    scientific_only = [dict(item) for item in DEMO_SCIENTIFIC_ONLY]
    authored_measurement = {
        **measurement,
        "known_inputs": known_inputs,
        "scientific_only_constructs": scientific_only,
    }
    grounded, status = measurement_structure_grounding(authored_measurement, latent)
    if grounded is None or status != "VALID":
        raise ValueError(f"Production measurement grounding rejected DEMO repair: {status}")
    measurement_artifact = MeasurementStructureContract.model_validate(grounded).model_dump(
        mode="json", by_alias=True
    )

    identification = check_identifiability(latent, measurement)
    identification_status = _canonical_identification_status(identification)
    causal_design = build_causal_design(
        latent,
        measurement,
        identification_status,
        known_inputs=known_inputs,
        scientific_only_constructs=scientific_only,
    )
    structural_plan = build_structural_plan(causal_design)
    latent_artifact = {"latent_structure": latent}
    causal_artifact = {
        "causal_design": causal_design.model_dump(mode="json", by_alias=True),
    }
    structural_plan_artifact = {
        "structural_plan": structural_plan.model_dump(mode="json", by_alias=True),
    }
    validation_artifact = _compact_demo_validation(stored_validation, DEMO_INDICATOR_ORDER)

    construct_by_name = {
        str(item["name"]): item for item in causal_artifact["causal_design"]["latent"]["constructs"]
    }
    state_names = get_state_names(structural_plan)
    state_constructs = [construct_by_name[name] for name in state_names]
    executable_edges = get_edges(structural_plan)
    manifest_indicators = get_manifest_indicators(structural_plan)

    audits = validation_artifact["indicators"]
    profiles: dict[str, JsonObject] = {}
    for indicator in manifest_indicators:
        name = str(indicator["name"])
        audit = audits[name]
        profile = audit["profile"]
        if profile is None:
            if name not in ARTIFICIAL_PROFILE_OVERRIDES:
                raise ValueError(f"Missing explicit artificial profile override for {name}")
            profile = ARTIFICIAL_PROFILE_OVERRIDES[name]
        profiles[name] = dict(profile)

    identifiable = set(identification_status["identifiable_treatments"])
    simulatable_treatments = sorted(identifiable & set(state_names))
    return FixtureSources(
        latent_artifact=latent_artifact,
        measurement_artifact=measurement_artifact,
        causal_artifact=causal_artifact,
        structural_plan_artifact=structural_plan_artifact,
        validation_artifact=validation_artifact,
        structural_plan=structural_plan,
        state_constructs=state_constructs,
        executable_edges=executable_edges,
        manifest_indicators=manifest_indicators,
        profiles=profiles,
        simulatable_treatments=simulatable_treatments,
    )


def _validate_outputs(outputs: dict[Path, JsonObject], sources: FixtureSources) -> None:
    """Run the same strict models and compiler boundary used by production."""
    measurement = MeasurementStructureContract.model_validate(
        outputs[ARTIFACT_ROOT / "measurement_structure.json"]
    )
    causal_design = CausalDesign.model_validate(
        outputs[ARTIFACT_ROOT / "causal_design.json"]["causal_design"]
    )
    structural_plan = build_structural_plan(causal_design)
    if structural_plan != sources.structural_plan:
        raise ValueError("Regenerated StructuralPlan differs from the validated fixture projection")
    if outputs[ARTIFACT_ROOT / "latent_structure.json"][
        "latent_structure"
    ] != causal_design.latent.model_dump(mode="json", by_alias=True):
        raise ValueError("LatentStructure and CausalDesign latent projections differ")
    if measurement.measurement_structure != causal_design.measurement:
        raise ValueError("MeasurementStructure and CausalDesign measurement projections differ")
    if outputs[ARTIFACT_ROOT / "structural_plan.json"][
        "structural_plan"
    ] != structural_plan.model_dump(mode="json", by_alias=True):
        raise ValueError("Persisted StructuralPlan differs from the production-derived plan")
    ValidationReportContract.model_validate(outputs[ARTIFACT_ROOT / "validation_report.json"])

    model_payload = outputs[ARTIFACT_ROOT / "statistical_model_spec.json"]
    StatisticalModelSpecContract.model_validate(model_payload)
    validated_spec, semantic_errors = validate_statistical_model_spec_dict(
        model_payload["statistical_model_spec"],
        sources.manifest_indicators,
    )
    if validated_spec is None:
        raise ValueError(
            "Production model-spec semantic validation failed:\n" + "\n".join(semantic_errors)
        )

    package_logger = logging.getLogger("nof1_causal_lab")
    previous_level = package_logger.level
    package_logger.setLevel(logging.ERROR)
    try:
        assembly = validate_assembly(
            model_payload["statistical_model_spec"],
            model_payload["authored_priors"],
            sources.structural_plan,
        )
    finally:
        package_logger.setLevel(previous_level)
    if not assembly.compile_ok or assembly.compiled_ssm is None:
        raise ValueError(
            f"Production model compiler rejected DEMO completion: {assembly.compile_error}"
        )

    PosteriorContract.model_validate(outputs[ARTIFACT_ROOT / "posterior.json"])
    baseline = BaselineReportContract.model_validate(
        outputs[ARTIFACT_ROOT / "baseline_report.json"]
    )
    baseline_treatments = sorted(item.treatment for item in baseline.intervention_results)
    if baseline_treatments != sources.simulatable_treatments:
        raise ValueError("Baseline report treatments differ from fitted identifiable states")

    trace = outputs[TRACE_ROOT / "baseline_report.json"]
    tool_call_ids: set[str] = set()
    tool_result_ids: set[str] = set()
    for message in trace["messages"]:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            if function.get("name") != "simulate":
                continue
            SimulateScenarioInput.model_validate(json.loads(function["arguments"]))
            tool_call_ids.add(str(call["id"]))
        if message.get("tool_name") == "simulate":
            SimulateScenarioToolResultContract.model_validate(json.loads(message["tool_result"]))
            tool_result_ids.add(str(message["tool_call_id"]))
    if tool_call_ids != tool_result_ids or len(tool_call_ids) != 5:
        raise ValueError("Baseline trace must contain five matched production-valid simulations")


def _build_outputs() -> dict[Path, JsonObject]:
    sources = _load_sources()
    panel_windows, n_timesteps, final_time = _panel_windows(
        [str(item["name"]) for item in sources.manifest_indicators]
    )
    model_spec, _edge_means = _build_statistical_model_spec(
        sources.state_constructs,
        sources.executable_edges,
        sources.manifest_indicators,
        sources.profiles,
    )
    posterior = _build_posterior(
        sources.manifest_indicators,
        sources.profiles,
        model_spec,
        panel_windows,
        n_timesteps,
    )
    baseline_report = _build_baseline_report(sources.simulatable_treatments)
    model_spec_trace = _build_model_spec_trace(
        sources.state_constructs,
        sources.manifest_indicators,
        model_spec,
    )
    baseline_trace = _build_baseline_trace(
        sources.state_constructs,
        sources.executable_edges,
        n_timesteps - 1,
        final_time,
    )
    outputs = {
        ARTIFACT_ROOT / "latent_structure.json": sources.latent_artifact,
        ARTIFACT_ROOT / "measurement_structure.json": sources.measurement_artifact,
        ARTIFACT_ROOT / "causal_design.json": sources.causal_artifact,
        ARTIFACT_ROOT / "structural_plan.json": sources.structural_plan_artifact,
        ARTIFACT_ROOT / "validation_report.json": sources.validation_artifact,
        ARTIFACT_ROOT / "statistical_model_spec.json": model_spec,
        ARTIFACT_ROOT / "posterior.json": posterior,
        ARTIFACT_ROOT / "baseline_report.json": baseline_report,
        TRACE_ROOT / "statistical_model_spec.json": model_spec_trace,
        TRACE_ROOT / "baseline_report.json": baseline_trace,
    }
    _validate_outputs(outputs, sources)
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail when checked-in artificial projections differ from deterministic generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = _build_outputs()
    changed: list[Path] = []
    for path, payload in outputs.items():
        expected = _json_bytes(payload)
        current = path.read_bytes() if path.exists() else None
        if current == expected:
            continue
        changed.append(path)
        if not args.check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)

    if args.check and changed:
        relative = "\n".join(f"  - {path.relative_to(REPO_ROOT)}" for path in changed)
        raise SystemExit(
            "Artificial DEMO completion is stale; run `bun run fixture:complete-demo`.\n" + relative
        )

    action = "verified" if args.check else "generated"
    print(f"{action} {len(outputs)} production-shaped DEMO fixture files")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, TypeError, ValueError) as exc:
        print(f"fixture completion failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
