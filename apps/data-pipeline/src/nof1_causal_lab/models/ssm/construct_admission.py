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
partial sub-DAG compiles fine — the ``n_manifest >= n_latent`` rank guard lives
only on the semantic-compile path used without a causal_design, which this module
does not take, so latent-only constructs (no indicator yet) still admit.

The verdict (admit / revise / accept) comes from :func:`reachability.
stage_outcome`; there is no status enum stored on any artifact.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import networkx as nx
import numpy as np

from nof1_causal_lab.artifacts.statistical_model_spec import (
    LikelihoodSpec,
    LinkFunction,
    ParameterSpec,
    StatisticalModelSpec,
)
from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_statistical_model_spec
from nof1_causal_lab.models.ssm.dynamics.spec import (
    HillEdgeSpec,
    LinearEdgeSpec,
    NodePotentialSpec,
)
from nof1_causal_lab.models.ssm.parameterization import build_prior_runtime_bundle
from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
    sample_prior_predictive_from_runtime,
)
from nof1_causal_lab.models.ssm.reachability import (
    CheckResult,
    check_confinement,
    check_coverage,
    check_edge_share,
    check_resolvability,
    check_saturation,
    check_scale,
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
class AdmissionReport:
    """Result of attempting to admit one construct."""

    name: str
    results: tuple[CheckResult, ...]
    outcome: str
    annotations: tuple[str, ...]
    admitted: bool


# --------------------------------------------------------------------------- #
# Planning + causal_design restriction
# --------------------------------------------------------------------------- #


def build_construct_order(causal_design: dict) -> list[str]:
    """Construct order (parents before children) along the causal arrows.

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
    scc_index = {
        node: min(order_index[member] for member in data["members"])
        for node, data in condensation.nodes(data=True)
    }
    order: list[str] = []
    for node in nx.lexicographical_topological_sort(condensation, key=scc_index.get):
        order.extend(sorted(condensation.nodes[node]["members"], key=order_index.get))
    return order


def restrict_causal_design(causal_design: dict, keep: set[str]) -> dict:
    """Restrict a causal_design to the subset ``keep`` of constructs.

    Every construct-indexed surface is filtered consistently so the partial model
    compiles: constructs, edges (both endpoints kept), indicators, and the
    estimation layout (state_order, edges, induced_dependencies, known_inputs).
    """
    spec = copy.deepcopy(causal_design)
    latent = spec.get("latent", {})
    latent["constructs"] = [c for c in latent.get("constructs", []) if c["name"] in keep]
    latent["edges"] = [
        e for e in latent.get("edges", []) if e.get("cause") in keep and e.get("effect") in keep
    ]
    measurement = spec.get("measurement", {})
    measurement["indicators"] = [
        i for i in measurement.get("indicators", []) if i.get("construct_name") in keep
    ]
    estimation = spec.get("estimation", {})
    estimation["state_order"] = [n for n in estimation.get("state_order", []) if n in keep]
    estimation["edges"] = [
        e for e in estimation.get("edges", []) if e.get("cause") in keep and e.get("effect") in keep
    ]
    estimation["induced_dependencies"] = [
        d for d in estimation.get("induced_dependencies", []) if set(d.get("between", [])) <= keep
    ]
    if "known_inputs" in estimation:
        estimation["known_inputs"] = [
            k for k in estimation.get("known_inputs", []) if k.get("construct") in keep
        ]
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


def _signal_from_linear_predictor(link: LinkFunction, lp: np.ndarray) -> np.ndarray:
    """Noise-free emission mean in data space from the linear predictor."""
    _lp = np.asarray(lp)
    if link == LinkFunction.IDENTITY:
        return _lp
    if link == LinkFunction.LOG:
        return np.exp(np.clip(_lp, -20.0, 20.0))
    if link == LinkFunction.LOGIT:
        return 1.0 / (1.0 + np.exp(-_lp))
    raise ValueError(f"unsupported link for signal extraction: {link}")


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
    """

    t_grid: jnp.ndarray
    obs_index_by_indicator: Mapping[str, np.ndarray]
    values_by_indicator: Mapping[str, np.ndarray]
    cadence: float
    span: float
    n_draws: int = 200
    seed: int = 0
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
    accepted: Mapping[str, str] | None = None,
) -> tuple[AdmissionState, AdmissionReport]:
    """Attempt to admit one construct; run the battery on the cumulative model."""
    trial = trial_admission_state(state, contribution)
    spec, registry = _compile_partial(trial, causal_design)
    pred = _sample_partial(spec, registry, design)
    results = _run_battery(spec, pred, design, contribution)

    outcome, annotations = stage_outcome(results, dict(accepted or {}))
    admitted = outcome.startswith("ADMITTED")
    report = AdmissionReport(
        name=contribution.name,
        results=tuple(results),
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


def _run_battery(
    spec: SSMSpec, pred: dict, design: DesignInfo, target: ConstructContribution
) -> list[CheckResult]:
    """Run the reachability battery on ``target``'s latent trajectory in a compiled model.

    Generic over the construct being measured: admission runs it on the construct being
    admitted; :func:`recheck_member` runs it on an already-admitted cycle member against the
    closed-loop model, where ``target.edge_parents`` now include the just-closed feedback edge.
    """
    latent_names, manifest_names, manifest_links = _spec_names(spec)
    d = latent_names.index(target.name)
    x = np.asarray(pred["latents"][:, :, d])
    dt = float(np.median(np.diff(np.asarray(design.t_grid))))
    pooled = design.pooled_obs_index

    results: list[CheckResult] = list(check_confinement(target.name, x, dt))

    # C2 scale anchor + C5 coverage come from the construct's indicator(s).
    anchor, anchor_src, anchor_detail = _scale_anchor(
        manifest_names, manifest_links, target, design.values_by_indicator
    )
    results.append(check_scale(target.name, x, anchor, anchor_src, anchor_detail))

    # C3 resolvability (only for dynamic constructs that own a potential well).
    comp_idx = _node_potential_index(spec, d)
    if comp_idx is not None:
        decay_site = f"vf_{comp_idx}_decay"
        if decay_site in pred:
            tau = 1.0 / np.asarray(pred[decay_site])
            results.append(check_resolvability(target.name, tau, design.cadence, design.span))

    # C4b edge overwhelm (edge-off re-simulation holds all else fixed).
    if target.edge_parents:
        edge_sites = _incoming_edge_sites(spec, target, latent_names, d)
        x_off = _resimulate_edge_off(
            spec, pred, design.t_grid, edge_sites, design.seed, design.transition_inputs
        )[:, :, d]
        results.extend(check_edge_share(target.name, x[:, pooled], np.asarray(x_off)[:, pooled]))

    # C4c Hill saturation (per saturating parent).
    for parent in target.hill_parents:
        p_idx = latent_names.index(parent)
        ec50_comp = _hill_ec50_index(spec, p_idx, d)
        ec50_site = f"vf_{ec50_comp}_EC50"
        if ec50_comp is not None and ec50_site in pred:
            parent_vals = np.asarray(pred["latents"][:, pooled, p_idx])
            results.append(
                check_saturation(
                    f"{parent}->{target.name}",
                    np.asarray(pred[ec50_site]),
                    parent_vals,
                )
            )

    # C5a/C5b/C5c coverage per indicator of this construct (per-indicator obs grid).
    for lik in target.likelihoods:
        var = lik.variable
        m = manifest_names.index(var)
        oi = np.asarray(design.obs_index_by_indicator[var])
        lp = np.asarray(pred["linear_predictors"][:, oi, m])
        pp_y = np.asarray(pred["observations"][:, oi, m])
        signal = _signal_from_linear_predictor(manifest_links[m], lp)
        results.extend(
            check_coverage(var, pp_y, signal, np.asarray(design.values_by_indicator[var]))
        )

    return results


def recheck_member(
    state: AdmissionState,
    target: ConstructContribution,
    causal_design: dict,
    design: DesignInfo,
) -> tuple[CheckResult, ...]:
    """Re-run the battery on an already-admitted member against the closed-loop model.

    ``state`` is the cumulative state *after* a feedback loop closed (it already contains
    ``target`` and the closing edge), and ``target`` carries the member's closed-loop edge
    set (``edge_parents`` now include the feedback source). Informational: the caller
    surfaces the results as a coupled recheck; they do not gate the admission.
    """
    spec, registry = _compile_partial(state, causal_design)
    pred = _sample_partial(spec, registry, design)
    return tuple(_run_battery(spec, pred, design, target))


def _scale_anchor(
    manifest_names: list[str],
    manifest_links: list[LinkFunction],
    contribution: ConstructContribution,
    data: Mapping[str, np.ndarray],
) -> tuple[float, str, str]:
    """Data-implied latent-scale anchor from the construct's reference indicator.

    The reference indicator has a fixed unit loading, so the anchor is its
    inverse-link IQR / 1.349 (the loading drops out). Latent-only constructs use
    the convention anchor 1.0.
    """
    if not contribution.likelihoods:
        return 1.0, "convention: no indicator", "convention anchor 1.0 — no indicator to anchor to"
    ref = contribution.likelihoods[0]
    m = manifest_names.index(ref.variable)
    link = manifest_links[m]
    q75, q25 = np.percentile(np.asarray(data[ref.variable]), [75, 25])
    iqr_xi = abs(float(_inverse_link_scalar(link, q75) - _inverse_link_scalar(link, q25)))
    anchor = iqr_xi / 1.349
    return (
        anchor,
        f"data via {ref.variable} (inverse-link IQR)",
        f"anchor {anchor:.2f} = {ref.variable} inverse-link IQR {iqr_xi:.2f} / 1.349 "
        "(reference indicator, unit loading)",
    )


def _inverse_link_scalar(link: LinkFunction, y: float) -> float:
    if link == LinkFunction.IDENTITY:
        return float(y)
    if link == LinkFunction.LOG:
        return float(np.log(max(y, 0.5)))
    if link == LinkFunction.LOGIT:
        p = float(np.clip(y, 1e-3, 1 - 1e-3))
        return float(np.log(p / (1.0 - p)))
    raise ValueError(f"unsupported link for anchor: {link}")


def _incoming_edge_sites(
    spec: SSMSpec,
    contribution: ConstructContribution,
    latent_names: list[str],
    target: int,
) -> list[str]:
    """Sample-site names of this construct's incoming edge strengths (to zero)."""
    sites: list[str] = []
    for parent in contribution.edge_parents:
        p_idx = latent_names.index(parent)
        found = _edge_component_index(spec, p_idx, target)
        if found is not None:
            comp_idx, suffix = found
            sites.append(f"vf_{comp_idx}_{suffix}")
    return sites


def _resimulate_edge_off(
    spec: SSMSpec,
    pred: dict,
    t_grid: jnp.ndarray,
    edge_sites: list[str],
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
    for site in edge_sites:
        if site in samples:
            samples[site] = jnp.zeros_like(jnp.asarray(samples[site]))
    latents, _linear_predictors = _simulate_vector_field_predictive_latents(
        spec, samples, t_grid, transition_inputs=transition_inputs, seed=seed
    )
    return latents


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #


def run_construct_build(
    causal_design: dict,
    contributions: Mapping[str, ConstructContribution],
    design: DesignInfo,
    accepted: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[AdmissionState, list[AdmissionReport]]:
    """Admit every construct in topological order; stop at the first non-admit.

    Returns the accumulated :class:`AdmissionState` and the per-construct reports.
    A construct that is BLOCKED (hard failure) or NEEDS DECISION (unaccepted soft
    failure) halts the build — the caller revises its contribution (or accepts the
    consequence via ``accepted``) and re-runs. The final ``AdmissionState`` yields
    the StatisticalModelSpec + priors to compile once every construct is admitted.
    """
    accepted = accepted or {}
    order = build_construct_order(causal_design)
    state = AdmissionState()
    reports: list[AdmissionReport] = []
    for name in order:
        state, report = admit_construct(
            state, contributions[name], causal_design, design, accepted.get(name)
        )
        reports.append(report)
        if not report.admitted:
            break
    return state, reports
