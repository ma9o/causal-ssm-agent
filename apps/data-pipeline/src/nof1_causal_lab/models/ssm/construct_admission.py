"""Gradual construct admission: build a model one construct at a time, gated by
the reachability battery on the cumulative partial model.

Constructs are admitted along the causal DAG's topological order (parents before
children). Each admission bundles the construct's contribution to the model —
its self-dynamics parameters, its incoming edges (from already-admitted
parents), and its emission(s) — into the growing :class:`~nof1_causal_lab.
artifacts.StatisticalModelSpec`, compiles the *cumulative partial* model, runs the **exact**
prior predictive (Diffrax over the true nonlinear drift, real emission
families), and feeds the resulting arrays to the reachability battery
(:mod:`nof1_causal_lab.models.ssm.reachability`).

Nothing here linearizes: the partial model is compiled and simulated through the
same exact engine the fit uses (``sample_prior_predictive_from_runtime``). A
partial sub-DAG compiles fine as long as every retained estimation state keeps
measurement support and the cumulative loading matrix can reach full column
rank.

The verdict (admit / revise / accept) comes from :func:`reachability.
stage_outcome`; there is no status enum stored on any artifact.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from statistics import NormalDist
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import networkx as nx
import numpy as np

from nof1_causal_lab.artifacts.statistical_model_spec import (
    LikelihoodSpec,
    LinkFunction,
    ParameterSpec,
    StatisticalModelSpec,
)
from nof1_causal_lab.distributions import DistributionFamily
from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_statistical_model_spec
from nof1_causal_lab.models.ssm.dynamics.spec import (
    HillEdgeSpec,
    LinearEdgeSpec,
    NodePotentialSpec,
)
from nof1_causal_lab.models.ssm.parameterization import build_prior_runtime_bundle
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    predictive_keys,
    sample_prior_predictive_from_runtime,
)
from nof1_causal_lab.models.ssm.reachability import (
    C1B_GROWTH_RATIO,
    C1B_MAX_EXPLOSIVE_FRAC,
    CheckResult,
    check_confinement,
    check_coverage,
    check_data_availability,
    check_edge_share,
    check_resolvability,
    check_saturation,
    check_scale,
    check_transmission,
    stage_outcome,
)
from nof1_causal_lab.utils.causal_design import (
    get_estimation_edges,
    get_estimation_state_order,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from nof1_causal_lab.models.ssm.model import SSMSpec
    from nof1_causal_lab.models.ssm.priors import PriorRegistry

# --------------------------------------------------------------------------- #
# Proposal / accumulation types
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConstructContribution:
    """One construct's contribution to the growing model.

    ``likelihoods``/``parameters``/``priors`` are canonical StatisticalModelSpec fragments
    (the parameter names must match the semantic-binding contract, e.g.
    ``rho_<c>``, ``self_limit_<c>``, ``beta_<p>_<c>``, ``hill_emax_<p>_<c>``,
    ``lambda_<ind>_<c>``, ``obs_sd_<ind>``). ``edge_parents`` and ``hill_parents``
    name the incoming parents used by the edge-overwhelm (C4b) and Hill
    saturation (C4c) checks.
    """

    name: str
    likelihoods: tuple[LikelihoodSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    priors: Mapping[str, dict] = field(default_factory=dict)
    edge_parents: tuple[str, ...] = ()
    hill_parents: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionState:
    """Accumulated admitted constructs and the model fragments they contribute."""

    names: tuple[str, ...] = ()
    likelihoods: tuple[LikelihoodSpec, ...] = ()
    parameters: tuple[ParameterSpec, ...] = ()
    priors: Mapping[str, dict] = field(default_factory=dict)
    annotations: tuple[str, ...] = ()

    def statistical_model_spec(self) -> StatisticalModelSpec:
        return StatisticalModelSpec(
            likelihoods=list(self.likelihoods), parameters=list(self.parameters)
        )


@dataclass(frozen=True)
class AdmissionTiming:
    """One measured phase of a construct-admission check."""

    phase: str
    label: str
    duration_ms: float
    checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdmissionReport:
    """Result of attempting to admit one construct."""

    name: str
    results: tuple[CheckResult, ...]
    timings: tuple[AdmissionTiming, ...]
    outcome: str
    annotations: tuple[str, ...]
    admitted: bool


@dataclass(frozen=True)
class ConstructAdmissionUnit:
    """One schedulable component of the construct graph.

    Singleton components may run in parallel once their predecessor components
    have been admitted.  Members of a lagged-feedback component stay in
    ``state_order`` and are admitted sequentially inside the component.
    """

    unit_id: str
    constructs: tuple[str, ...]
    predecessors: tuple[str, ...]


@dataclass(frozen=True)
class FullAdmissionValidation:
    """One exact shared simulation plus per-construct full-model reports."""

    reports: tuple[AdmissionReport, ...]
    timings: tuple[AdmissionTiming, ...]


# --------------------------------------------------------------------------- #
# Planning + causal_design restriction
# --------------------------------------------------------------------------- #


def build_construct_units(causal_design: dict) -> list[ConstructAdmissionUnit]:
    """Build the SCC-condensed fork/join plan for construct admission.

    The universe is the estimation projection's ``state_order`` — constructs
    measurement-structure marginalized, anchored, or dropped out of estimation carry no
    state, so there is nothing to admit for them (restricting the spec to one
    would fail compilation with an empty state_order).

    Ties (independent roots) break by the state_order position for
    determinism. Time-invariant confounders, being edge sources, naturally sort
    first. Lagged feedback loops are legal latent structure (the latent-structure
    validator only forbids *contemporaneous* cycles), so the sort runs on the
    condensation: members of a feedback cycle are admitted back-to-back in
    state_order, and restrict_causal_design defers the closing edge until
    the whole cycle is admitted.
    """
    constructs = get_estimation_state_order(causal_design)
    order_index = {name: i for i, name in enumerate(constructs)}
    graph = nx.DiGraph()
    graph.add_nodes_from(constructs)
    for edge in get_estimation_edges(causal_design):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if cause in order_index and effect in order_index:
            graph.add_edge(cause, effect)
    condensation = nx.condensation(graph)
    component_members = {
        node: tuple(
            sorted(
                (str(member) for member in data["members"]),
                key=lambda member: order_index[member],
            )
        )
        for node, data in condensation.nodes(data=True)
    }
    scc_index = {node: order_index[members[0]] for node, members in component_members.items()}
    component_order = list(
        nx.lexicographical_topological_sort(
            condensation,
            key=lambda node: scc_index[node],
        )
    )
    unit_id_by_component = {
        node: (
            component_members[node][0]
            if len(component_members[node]) == 1
            else f"feedback:{component_members[node][0]}"
        )
        for node in component_order
    }
    return [
        ConstructAdmissionUnit(
            unit_id=unit_id_by_component[node],
            constructs=component_members[node],
            predecessors=tuple(
                unit_id_by_component[parent]
                for parent in sorted(
                    condensation.predecessors(node),
                    key=lambda component: scc_index[component],
                )
            ),
        )
        for node in component_order
    ]


def build_construct_order(causal_design: dict) -> list[str]:
    """Deterministic flattened order used for assembly and stable presentation."""
    return [
        construct for unit in build_construct_units(causal_design) for construct in unit.constructs
    ]


def restrict_causal_design(causal_design: dict, keep: set[str]) -> dict:
    """Restrict a causal_design to the subset ``keep`` of constructs.

    Every construct-indexed surface is filtered consistently so the partial model
    compiles. Known inputs feeding a retained state are dependencies of that state,
    not admission units, so their declaration, theoretical construct, source
    indicator, and incoming edge remain in the restricted design.
    """
    spec = copy.deepcopy(causal_design)
    estimation = spec.get("estimation", {})
    all_known_inputs = list(estimation.get("known_inputs", []))
    known_input_names = {
        str(item.get("construct") or item.get("construct_name"))
        for item in all_known_inputs
        if item.get("construct") or item.get("construct_name")
    }
    relevant_estimation_edges = [
        edge
        for edge in estimation.get("edges", [])
        if edge.get("effect") in keep and edge.get("cause") in (keep | known_input_names)
    ]
    relevant_input_names = {
        str(edge.get("cause"))
        for edge in relevant_estimation_edges
        if edge.get("cause") in known_input_names
    }
    relevant_known_inputs = [
        item
        for item in all_known_inputs
        if (item.get("construct") or item.get("construct_name")) in relevant_input_names
    ]
    input_source_indicators = {
        str(item["source_indicator"])
        for item in relevant_known_inputs
        if item.get("source_indicator")
    }

    latent = spec.get("latent", {})
    retained_construct_names = keep | relevant_input_names
    latent["constructs"] = [
        c for c in latent.get("constructs", []) if c["name"] in retained_construct_names
    ]
    latent["edges"] = [
        e
        for e in latent.get("edges", [])
        if e.get("cause") in retained_construct_names and e.get("effect") in keep
    ]
    measurement = spec.get("measurement", {})
    measurement["indicators"] = [
        i
        for i in measurement.get("indicators", [])
        if i.get("construct_name") in keep or i.get("name") in input_source_indicators
    ]
    estimation["state_order"] = [n for n in estimation.get("state_order", []) if n in keep]
    estimation["edges"] = relevant_estimation_edges
    estimation["induced_dependencies"] = [
        d for d in estimation.get("induced_dependencies", []) if set(d.get("between", [])) <= keep
    ]
    estimation["known_inputs"] = relevant_known_inputs
    return spec


# --------------------------------------------------------------------------- #
# Cumulative-model compilation + exact prior predictive
# --------------------------------------------------------------------------- #


def _compile_partial(
    state: AdmissionState,
    causal_design: dict,
) -> tuple[SSMSpec, PriorRegistry]:
    """Compile the cumulative partial model to an SSMSpec + prior registry."""
    restricted = restrict_causal_design(causal_design, set(state.names))
    spec, registry, _bindings, _diagnostics, _edge_lag = (
        compile_ssm_inputs_from_statistical_model_spec(
            state.statistical_model_spec(), dict(state.priors), causal_design=restricted
        )
    )
    return spec, registry


def _spec_names(spec: SSMSpec) -> tuple[list[str], list[str], list[LinkFunction]]:
    """Latent names, manifest names, manifest links of a compiled spec.

    These are always populated after compilation; a missing one is a compiler
    invariant violation, not a recoverable case.
    """
    if spec.latent_names is None or spec.manifest_names is None or spec.manifest_links is None:
        raise ValueError("compiled SSMSpec is missing latent/manifest metadata")
    return spec.latent_names, spec.manifest_names, spec.manifest_links


def _node_potential_index(spec: SSMSpec, latent_idx: int) -> int | None:
    """Component index of the NodePotential owning ``latent_idx`` (its site prefix)."""
    for comp_idx, comp in enumerate(spec.dynamics_spec.components):
        if isinstance(comp, NodePotentialSpec) and comp.target == latent_idx:
            return comp_idx
    return None


def _edge_component_index(spec: SSMSpec, source: int, target: int) -> tuple[int, str] | None:
    """Component index + weight/Emax site suffix of the edge (source -> target)."""
    for comp_idx, comp in enumerate(spec.dynamics_spec.components):
        if isinstance(comp, LinearEdgeSpec) and comp.source == source and comp.target == target:
            return comp_idx, "weight"
        if isinstance(comp, HillEdgeSpec) and comp.source == source and comp.target == target:
            return comp_idx, "Emax"
    return None


def _hill_ec50_index(spec: SSMSpec, source: int, target: int) -> int | None:
    for comp_idx, comp in enumerate(spec.dynamics_spec.components):
        if isinstance(comp, HillEdgeSpec) and comp.source == source and comp.target == target:
            return comp_idx
    return None


def _signal_from_linear_predictor(
    link: LinkFunction,
    lp: np.ndarray,
    *,
    spec: SSMSpec,
    pred: Mapping[str, Any],
    manifest_index: int,
) -> np.ndarray:
    """Noise-free emission mean in data space from the linear predictor."""
    _lp = np.asarray(lp)
    if link == LinkFunction.IDENTITY:
        return _lp
    if link == LinkFunction.LOG:
        return np.exp(np.clip(_lp, -20.0, 20.0))
    if link == LinkFunction.LOGIT:
        return 1.0 / (1.0 + np.exp(-_lp))
    if link == LinkFunction.PROBIT:
        cdf = np.vectorize(NormalDist().cdf, otypes=[float])
        return cdf(_lp)
    if link == LinkFunction.INVERSE:
        return 1.0 / np.clip(_lp, 1e-6, None)

    if spec.manifest_level_counts is None:
        raise ValueError(f"{link.value} signal extraction requires manifest level counts")
    level_count = int(spec.manifest_level_counts[manifest_index])
    if level_count < 2:
        raise ValueError(f"{link.value} signal extraction requires at least two declared levels")

    if link == LinkFunction.CUMULATIVE_LOGIT:
        base = np.asarray(pred["obs_ordered_base"])[:, manifest_index]
        if level_count > 2:
            gaps = np.asarray(pred["obs_ordered_gaps"])[:, manifest_index, : level_count - 2]
            cutpoints = np.concatenate(
                [base[:, None], base[:, None] + np.cumsum(gaps, axis=1)],
                axis=1,
            )
        else:
            cutpoints = base[:, None]
        mid_cdf = 1.0 / (
            1.0 + np.exp(-np.clip(cutpoints[:, None, :] - _lp[:, :, None], -30.0, 30.0))
        )
        cdf = np.concatenate(
            [
                np.zeros((*_lp.shape, 1)),
                mid_cdf,
                np.ones((*_lp.shape, 1)),
            ],
            axis=2,
        )
        return np.diff(cdf, axis=2)

    if link == LinkFunction.SOFTMAX:
        intercepts = np.asarray(pred["obs_cat_intercepts"])[:, manifest_index, : level_count - 1]
        slopes = np.asarray(pred["obs_cat_slopes"])[:, manifest_index, : level_count - 1]
        nonbaseline = intercepts[:, None, :] + slopes[:, None, :] * _lp[:, :, None]
        logits = np.concatenate([np.zeros((*_lp.shape, 1)), nonbaseline], axis=2)
        logits = logits - np.max(logits, axis=2, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=2, keepdims=True)
        return probabilities
    raise ValueError(f"unsupported link for signal extraction: {link}")


def _draw_scalar_parameter(pred: Mapping[str, Any], name: str, n_draws: int) -> np.ndarray:
    """Return one scalar likelihood hyperparameter per prior draw."""
    values = np.asarray(pred[name], dtype=float)
    if values.shape[0] != n_draws or values.size != n_draws:
        raise ValueError(f"predictive parameter {name!r} must be scalar per draw")
    return values.reshape(n_draws, 1)


def _conditional_variance_for_signal(
    distribution: DistributionFamily,
    signal: np.ndarray,
    pred: Mapping[str, Any],
    manifest_index: int,
) -> np.ndarray:
    """Exact family variance around a scalar prior-predictive emission mean."""
    mean = np.asarray(signal, dtype=float)
    if mean.ndim != 2:
        raise ValueError("scalar conditional variance requires draws by observed-time means")
    n_draws = mean.shape[0]

    if distribution in {DistributionFamily.GAUSSIAN, DistributionFamily.STUDENT_T}:
        manifest_cov = np.asarray(pred["manifest_cov"], dtype=float)
        if manifest_cov.shape[0] != n_draws or manifest_cov.ndim != 3:
            raise ValueError("manifest_cov must contain one covariance matrix per draw")
        variance = manifest_cov[:, manifest_index, manifest_index][:, None]
        if distribution == DistributionFamily.STUDENT_T:
            df = _draw_scalar_parameter(pred, "obs_df", n_draws)
            factor = np.full_like(df, np.inf)
            np.divide(df, df - 2.0, out=factor, where=df > 2.0)
            variance = variance * factor
        return np.broadcast_to(variance, mean.shape)

    if distribution == DistributionFamily.POISSON:
        return np.maximum(mean, 1e-8)
    if distribution == DistributionFamily.GAMMA:
        shape = _draw_scalar_parameter(pred, "obs_shape", n_draws)
        return np.maximum(mean, 1e-8) ** 2 / (shape + 1e-8)
    if distribution == DistributionFamily.BERNOULLI:
        probability = np.clip(mean, 1e-7, 1.0 - 1e-7)
        return probability * (1.0 - probability)
    if distribution == DistributionFamily.NEGATIVE_BINOMIAL:
        dispersion = _draw_scalar_parameter(pred, "obs_r", n_draws)
        count_mean = np.maximum(mean, 1e-8)
        return count_mean + count_mean**2 / (dispersion + 1e-8)
    if distribution == DistributionFamily.BETA:
        concentration = _draw_scalar_parameter(pred, "obs_concentration", n_draws)
        probability = np.clip(mean, 1e-7, 1.0 - 1e-7)
        return probability * (1.0 - probability) / (concentration + 1.0)
    raise ValueError(f"{distribution.value} uses probability-vector transmission")


# --------------------------------------------------------------------------- #
# Admission
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DesignInfo:
    """Sampling design + observed data needed by the checks, bound once per build.

    Real longitudinal data is irregular and per-indicator, so observations are
    indexed per indicator: ``obs_index_by_indicator`` maps each indicator to the
    ``t_grid`` indices where it was actually observed, and ``values_by_indicator``
    holds the aligned observed values. ``observation_support`` and
    ``transition_inputs`` are threaded to the exact prior-predictive sampler and the
    edge-off re-simulation when present (real data); synthetic tests leave them
    ``None`` and share a single index across indicators.

    ``c1b_growth_ratio`` / ``c1b_max_explosive_frac`` calibrate C1b confinement: a
    draw explodes when its late-window amplitude exceeds ``c1b_growth_ratio`` times
    its own early amplitude, and the check reds when at least
    ``c1b_max_explosive_frac`` of draws do. The defaults encode the model class's
    confinement commitment (every self-dynamics component is a restoring force);
    they are design calibration, not part of the statistic — an intrinsically
    trending domain raises them here instead of accepting a standing soft-fail.
    """

    t_grid: jnp.ndarray
    obs_index_by_indicator: Mapping[str, np.ndarray]
    values_by_indicator: Mapping[str, np.ndarray]
    n_draws: int = 200
    seed: int = 0
    c1b_growth_ratio: float = C1B_GROWTH_RATIO
    c1b_max_explosive_frac: float = C1B_MAX_EXPLOSIVE_FRAC
    observation_support: Any = None
    transition_inputs: Any = None

    @property
    def pooled_obs_index(self) -> np.ndarray:
        """Sorted union of every indicator's observation indices.

        Used by the latent-at-observation checks (edge overwhelm, saturation) that
        are indicator-agnostic — they read the latent path where any data exists.
        """
        if not self.obs_index_by_indicator:
            return np.arange(int(np.asarray(self.t_grid).shape[0]))
        return np.unique(
            np.concatenate([np.asarray(v) for v in self.obs_index_by_indicator.values()])
        )

    def observation_indices_for(self, indicators: tuple[str, ...]) -> np.ndarray:
        """Sorted union of actual observation indices for the requested indicators."""
        indices = [
            np.asarray(self.obs_index_by_indicator[name])
            for name in indicators
            if name in self.obs_index_by_indicator
            and np.asarray(self.obs_index_by_indicator[name]).size > 0
        ]
        if not indices:
            return np.asarray([], dtype=int)
        return np.unique(np.concatenate(indices))


@dataclass(frozen=True)
class _EdgeOffTarget:
    """One compiled edge coordinate to disable under the same predictive draws."""

    vector_field_sites: tuple[str, ...] = ()
    input_effect_cells: tuple[tuple[int, int], ...] = ()


def trial_admission_state(
    state: AdmissionState, contribution: ConstructContribution
) -> AdmissionState:
    """The cumulative state that *would* result from admitting ``contribution``.

    Used both to run the battery (:func:`admit_construct`) and to derive the
    design (grid + observed data) against the same partial model.
    """
    return AdmissionState(
        names=(*state.names, contribution.name),
        likelihoods=(*state.likelihoods, *contribution.likelihoods),
        parameters=(*state.parameters, *contribution.parameters),
        priors={**dict(state.priors), **dict(contribution.priors)},
        annotations=state.annotations,
    )


def admit_construct(
    state: AdmissionState,
    contribution: ConstructContribution,
    causal_design: dict,
    design: DesignInfo,
    accepted: Mapping[tuple[str, str], str] | None = None,
) -> tuple[AdmissionState, AdmissionReport]:
    """Attempt to admit one construct; run the battery on the cumulative model."""
    trial = trial_admission_state(state, contribution)

    started = perf_counter_ns()
    spec, registry = _compile_partial(trial, causal_design)
    timings = [
        AdmissionTiming(
            phase="model_compilation",
            label="Model compilation",
            duration_ms=_elapsed_ms(started),
        )
    ]

    started = perf_counter_ns()
    pred = _sample_partial(spec, registry, design)
    jax.block_until_ready(pred)
    timings.append(
        AdmissionTiming(
            phase="prior_predictive",
            label="Exact prior-predictive simulation",
            duration_ms=_elapsed_ms(started),
        )
    )

    results, diagnostic_timings = _run_battery(spec, pred, design, contribution)
    timings.extend(diagnostic_timings)

    started = perf_counter_ns()
    outcome, annotations = stage_outcome(results, accepted or {})
    timings.append(
        AdmissionTiming(
            phase="admission_decision",
            label="Admission decision",
            duration_ms=_elapsed_ms(started),
        )
    )
    admitted = outcome.startswith("ADMITTED")
    report = AdmissionReport(
        name=contribution.name,
        results=tuple(results),
        timings=tuple(timings),
        outcome=outcome,
        annotations=annotations,
        admitted=admitted,
    )
    if not admitted:
        return state, report
    return replace(trial, annotations=(*state.annotations, *annotations)), report


def _sample_partial(spec: SSMSpec, registry: PriorRegistry, design: DesignInfo) -> dict:
    """Exact prior-predictive draws for a compiled partial model on the design grid."""
    bundle = build_prior_runtime_bundle(spec, registry)
    return sample_prior_predictive_from_runtime(
        spec,
        bundle,
        design.t_grid,
        observation_support=design.observation_support,
        transition_inputs=design.transition_inputs,
        num_samples=design.n_draws,
        seed=design.seed,
    )


def _elapsed_ms(started_ns: int) -> float:
    return (perf_counter_ns() - started_ns) / 1_000_000


def _run_battery(
    spec: SSMSpec, pred: dict, design: DesignInfo, target: ConstructContribution
) -> tuple[list[CheckResult], list[AdmissionTiming]]:
    """Run the reachability battery on ``target``'s latent trajectory in a compiled model.

    Generic over the construct being measured: admission runs it on the construct being
    admitted; :func:`recheck_member` runs it on an already-admitted cycle member against the
    closed-loop model, where ``target.edge_parents`` now include the just-closed feedback edge.
    """
    latent_names, manifest_names, manifest_links = _spec_names(spec)
    d = latent_names.index(target.name)
    x = np.asarray(pred["latents"][:, :, d])
    times = np.asarray(design.t_grid, dtype=float)
    indicator_names = tuple(lik.variable for lik in target.likelihoods)
    target_obs = design.observation_indices_for(indicator_names)
    structural_indices = target_obs if target_obs.size else np.arange(times.size)

    results: list[CheckResult] = []
    timings: list[AdmissionTiming] = []

    started = perf_counter_ns()
    phase_results = list(
        check_confinement(
            target.name,
            x,
            times,
            growth_ratio=design.c1b_growth_ratio,
            max_explosive_frac=design.c1b_max_explosive_frac,
        )
    )
    results.extend(phase_results)
    timings.append(
        AdmissionTiming(
            phase="c1_confinement",
            label="C1 confinement",
            duration_ms=_elapsed_ms(started),
            checks=tuple(result.check for result in phase_results),
        )
    )

    started = perf_counter_ns()
    result = check_scale(target.name, x)
    results.append(result)
    timings.append(
        AdmissionTiming(
            phase="c2_latent_scale",
            label="C2 latent scale",
            duration_ms=_elapsed_ms(started),
            checks=(result.check,),
        )
    )

    # C3 resolvability (only for dynamic constructs that own a potential well).
    comp_idx = _node_potential_index(spec, d)
    if comp_idx is not None:
        decay_site = f"vf_{comp_idx}_decay"
        if decay_site in pred:
            started = perf_counter_ns()
            tau = 1.0 / np.asarray(pred[decay_site])
            result = check_resolvability(target.name, tau, times[target_obs])
            results.append(result)
            timings.append(
                AdmissionTiming(
                    phase="c3_resolvability",
                    label="C3 resolvability",
                    duration_ms=_elapsed_ms(started),
                    checks=(result.check,),
                )
            )

    # C4b edge overwhelm (edge-off re-simulation holds all else fixed).
    for parent in target.edge_parents:
        started = perf_counter_ns()
        edge_target = _incoming_edge_off_target(
            spec, replace(target, edge_parents=(parent,)), latent_names, d
        )
        x_off = _resimulate_edge_off(
            spec, pred, design.t_grid, edge_target, design.seed, design.transition_inputs
        )[:, :, d]
        edge_label = f"{parent}->{target.name}"
        phase_results = list(
            check_edge_share(
                edge_label,
                x[:, structural_indices],
                np.asarray(x_off)[:, structural_indices],
            )
        )
        results.extend(phase_results)
        timings.append(
            AdmissionTiming(
                phase=f"c4b_edge_overwhelm:{edge_label}",
                label=f"C4b edge-off resimulation: {parent} → {target.name}",
                duration_ms=_elapsed_ms(started),
                checks=tuple(result.check for result in phase_results),
            )
        )

    # C4c Hill saturation (per saturating parent).
    for parent in target.hill_parents:
        p_idx = latent_names.index(parent)
        ec50_comp = _hill_ec50_index(spec, p_idx, d)
        ec50_site = f"vf_{ec50_comp}_EC50"
        if ec50_comp is not None and ec50_site in pred:
            started = perf_counter_ns()
            parent_vals = np.asarray(pred["latents"][:, structural_indices, p_idx])
            result = check_saturation(
                f"{parent}->{target.name}",
                np.asarray(pred[ec50_site]),
                np.asarray(pred[f"vf_{ec50_comp}_n"]),
                parent_vals,
            )
            results.append(result)
            timings.append(
                AdmissionTiming(
                    phase=f"c4c_saturation:{parent}->{target.name}",
                    label=f"C4c saturation: {parent} → {target.name}",
                    duration_ms=_elapsed_ms(started),
                    checks=(result.check,),
                )
            )

    time_invariant_mask = spec.diffusion_block.time_invariant_mask
    target_is_time_invariant = bool(
        time_invariant_mask is not None and np.asarray(time_invariant_mask, dtype=bool)[d]
    )

    # C5a/C5b coverage for every indicator; C5c transmission only for dynamic constructs.
    for lik in target.likelihoods:
        started = perf_counter_ns()
        var = lik.variable
        observed = np.asarray(design.values_by_indicator[var])
        if observed.size == 0:
            result = check_data_availability(var)
            results.append(result)
            timings.append(
                AdmissionTiming(
                    phase=f"c5_data_availability:{var}",
                    label=f"C5 data availability: {var}",
                    duration_ms=_elapsed_ms(started),
                    checks=(result.check,),
                )
            )
            continue
        m = manifest_names.index(var)
        oi = np.asarray(design.obs_index_by_indicator[var])
        pp_y = np.asarray(pred["observations"][:, oi, m])
        if manifest_links[m] in {LinkFunction.CUMULATIVE_LOGIT, LinkFunction.SOFTMAX}:
            lp = np.asarray(pred["linear_predictors"][:, oi, m])
            signal = _signal_from_linear_predictor(
                manifest_links[m],
                lp,
                spec=spec,
                pred=pred,
                manifest_index=m,
            )
        else:
            signal = np.asarray(pred["expected_observations"][:, oi, m])
        level_count = (
            int(spec.manifest_level_counts[m]) if spec.manifest_level_counts is not None else None
        )
        phase_results = list(
            check_coverage(
                var,
                pp_y,
                observed,
                distribution=lik.distribution.value,
                level_count=level_count,
            )
        )
        if not target_is_time_invariant:
            conditional_variance = (
                None
                if signal.ndim == 3
                else _conditional_variance_for_signal(
                    lik.distribution,
                    signal,
                    pred,
                    m,
                )
            )
            phase_results.append(check_transmission(var, signal, conditional_variance))
        results.extend(phase_results)
        timings.append(
            AdmissionTiming(
                phase=f"c5_coverage:{var}",
                label=f"C5 emission reachability: {var}",
                duration_ms=_elapsed_ms(started),
                checks=tuple(result.check for result in phase_results),
            )
        )

    return results, timings


def recheck_member(
    state: AdmissionState,
    target: ConstructContribution,
    causal_design: dict,
    design: DesignInfo,
) -> tuple[tuple[CheckResult, ...], tuple[AdmissionTiming, ...]]:
    """Re-run the battery on an already-admitted member against the closed-loop model.

    ``state`` is the cumulative state *after* a feedback loop closed (it already contains
    ``target`` and the closing edge), and ``target`` carries the member's closed-loop edge
    set (``edge_parents`` now include the feedback source). Informational: the caller
    surfaces the results as a coupled recheck; they do not gate the admission.
    """
    started = perf_counter_ns()
    spec, registry = _compile_partial(state, causal_design)
    timings = [
        AdmissionTiming(
            phase="model_compilation",
            label="Model compilation",
            duration_ms=_elapsed_ms(started),
        )
    ]
    started = perf_counter_ns()
    pred = _sample_partial(spec, registry, design)
    jax.block_until_ready(pred)
    timings.append(
        AdmissionTiming(
            phase="prior_predictive",
            label="Exact prior-predictive simulation",
            duration_ms=_elapsed_ms(started),
        )
    )
    results, diagnostic_timings = _run_battery(spec, pred, design, target)
    timings.extend(diagnostic_timings)
    return tuple(results), tuple(timings)


def validate_full_admission_state(
    state: AdmissionState,
    targets: tuple[ConstructContribution, ...],
    causal_design: dict,
    design: DesignInfo,
    accepted: Mapping[str, Mapping[tuple[str, str], str]] | None = None,
) -> FullAdmissionValidation:
    """Gate publication with one exact full-model simulation and all construct batteries."""
    started = perf_counter_ns()
    spec, registry = _compile_partial(state, causal_design)
    timings = [
        AdmissionTiming(
            phase="model_compilation",
            label="Full-model compilation",
            duration_ms=_elapsed_ms(started),
        )
    ]

    started = perf_counter_ns()
    pred = _sample_partial(spec, registry, design)
    jax.block_until_ready(pred)
    timings.append(
        AdmissionTiming(
            phase="prior_predictive",
            label="Exact full-model prior-predictive simulation",
            duration_ms=_elapsed_ms(started),
        )
    )

    accepted = accepted or {}
    reports: list[AdmissionReport] = []
    for target in targets:
        results, diagnostic_timings = _run_battery(spec, pred, design, target)
        outcome, annotations = stage_outcome(results, accepted.get(target.name, {}))
        reports.append(
            AdmissionReport(
                name=target.name,
                results=tuple(results),
                timings=tuple(diagnostic_timings),
                outcome=outcome,
                annotations=annotations,
                admitted=outcome.startswith("ADMITTED"),
            )
        )
    return FullAdmissionValidation(reports=tuple(reports), timings=tuple(timings))


def _incoming_edge_off_target(
    spec: SSMSpec,
    contribution: ConstructContribution,
    latent_names: list[str],
    target: int,
) -> _EdgeOffTarget:
    """Compiled vector-field sites or input-effect cells for incoming edges."""
    sites: list[str] = []
    input_cells: list[tuple[int, int]] = []
    input_names = list(spec.input_names or [])
    for parent in contribution.edge_parents:
        if parent in latent_names:
            p_idx = latent_names.index(parent)
            found = _edge_component_index(spec, p_idx, target)
            if found is not None:
                comp_idx, suffix = found
                sites.append(f"vf_{comp_idx}_{suffix}")
        elif parent in input_names:
            input_cells.append((target, input_names.index(parent)))
    if contribution.edge_parents and not sites and not input_cells:
        raise ValueError(
            "Could not resolve an edge-off coordinate for incoming parents "
            f"{list(contribution.edge_parents)!r}"
        )
    return _EdgeOffTarget(tuple(sites), tuple(input_cells))


def _resimulate_edge_off(
    spec: SSMSpec,
    pred: dict,
    t_grid: jnp.ndarray,
    edge_target: _EdgeOffTarget,
    seed: int,
    transition_inputs: Any = None,
) -> jnp.ndarray:
    """Re-simulate the latents with the given edge sites zeroed, all else fixed.

    Reuses the exact param draws (and Brownian path via ``seed``) from ``pred`` so
    the only difference from the edge-on trajectory is the zeroed edge — the
    same-noise contrast :func:`reachability.check_edge_share` expects. The design's
    ``transition_inputs`` are held identical to the edge-on run.
    """
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
        _simulate_vector_field_predictive_latents,
    )

    samples = dict(pred)
    for site in edge_target.vector_field_sites:
        if site in samples:
            samples[site] = jnp.zeros_like(jnp.asarray(samples[site]))
    if edge_target.input_effect_cells:
        input_effect = jnp.asarray(samples["input_effect"])
        if input_effect.ndim != 3:
            raise ValueError("predictive input_effect must have shape draws x states x inputs")
        for target_idx, input_idx in edge_target.input_effect_cells:
            input_effect = input_effect.at[:, target_idx, input_idx].set(0.0)
        samples["input_effect"] = input_effect
    latents, _linear_predictors = _simulate_vector_field_predictive_latents(
        spec,
        samples,
        t_grid,
        transition_inputs=transition_inputs,
        rng_key=predictive_keys(seed).latents,
    )
    return latents
