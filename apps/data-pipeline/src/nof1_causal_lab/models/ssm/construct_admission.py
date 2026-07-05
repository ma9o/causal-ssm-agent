"""Gradual construct admission: build a model one construct at a time, gated by
the reachability battery on the cumulative partial model.

Constructs are admitted along the causal DAG's topological order (parents before
children). Each admission bundles the construct's contribution to the model —
its self-dynamics parameters, its incoming edges (from already-admitted
parents), and its emission(s) — into the growing :class:`~nof1_causal_lab.
artifacts.ModelSpec`, compiles the *cumulative partial* model, runs the **exact**
prior predictive (Diffrax over the true nonlinear drift, real emission
families), and feeds the resulting arrays to the reachability battery
(:mod:`nof1_causal_lab.models.ssm.reachability`).

Nothing here linearizes: the partial model is compiled and simulated through the
same exact engine the fit uses (``sample_prior_predictive_from_runtime``). A
partial sub-DAG compiles fine — the ``n_manifest >= n_latent`` rank guard lives
only on the semantic-compile path used without a causal_spec, which this module
does not take, so latent-only constructs (no indicator yet) still admit.

The verdict (admit / revise / accept) comes from :func:`reachability.
stage_outcome`; there is no status enum stored on any artifact.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
import networkx as nx
import numpy as np

from nof1_causal_lab.artifacts.model_spec import (
    LikelihoodSpec,
    LinkFunction,
    ModelSpec,
    ParameterSpec,
)
from nof1_causal_lab.models.ssm.compile.inputs import compile_ssm_inputs_from_model_spec
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
from nof1_causal_lab.utils.causal_spec import (
    get_constructs,
    get_estimation_edges,
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

    ``likelihoods``/``parameters``/``priors`` are canonical ModelSpec fragments
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

    def model_spec(self) -> ModelSpec:
        return ModelSpec(likelihoods=list(self.likelihoods), parameters=list(self.parameters))


@dataclass(frozen=True)
class AdmissionReport:
    """Result of attempting to admit one construct."""

    name: str
    results: tuple[CheckResult, ...]
    outcome: str
    annotations: tuple[str, ...]
    admitted: bool


# --------------------------------------------------------------------------- #
# Planning + causal_spec restriction
# --------------------------------------------------------------------------- #


def build_construct_order(causal_spec: dict) -> list[str]:
    """Topological construct order (parents before children) along the causal DAG.

    Ties (independent roots) break by the causal_spec construct order for
    determinism. Time-invariant confounders, being edge sources, naturally sort
    first.
    """
    constructs = [c["name"] for c in get_constructs(causal_spec)]
    order_index = {name: i for i, name in enumerate(constructs)}
    graph = nx.DiGraph()
    graph.add_nodes_from(constructs)
    for edge in get_estimation_edges(causal_spec):
        cause = edge.get("cause") if isinstance(edge, dict) else edge.cause
        effect = edge.get("effect") if isinstance(edge, dict) else edge.effect
        if cause in order_index and effect in order_index:
            graph.add_edge(cause, effect)
    return list(nx.lexicographical_topological_sort(graph, key=order_index.get))


def restrict_causal_spec(causal_spec: dict, keep: set[str]) -> dict:
    """Restrict a causal_spec to the subset ``keep`` of constructs.

    Every construct-indexed surface is filtered consistently so the partial model
    compiles: constructs, edges (both endpoints kept), indicators, and the
    estimation layout (state_order, edges, induced_dependencies, known_inputs).
    """
    spec = copy.deepcopy(causal_spec)
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
    causal_spec: dict,
) -> tuple[SSMSpec, PriorRegistry]:
    """Compile the cumulative partial model to an SSMSpec + prior registry."""
    restricted = restrict_causal_spec(causal_spec, set(state.names))
    spec, registry, _bindings, _diagnostics, _edge_lag = compile_ssm_inputs_from_model_spec(
        state.model_spec(), dict(state.priors), causal_spec=restricted
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
    """Sampling design + data needed by the checks, bound once per build."""

    t_grid: jnp.ndarray
    obs_idx: np.ndarray
    cadence: float
    span: float
    data: Mapping[str, np.ndarray]
    n_draws: int = 200
    seed: int = 0


def admit_construct(
    state: AdmissionState,
    contribution: ConstructContribution,
    causal_spec: dict,
    design: DesignInfo,
    accepted: Mapping[str, str] | None = None,
) -> tuple[AdmissionState, AdmissionReport]:
    """Attempt to admit one construct; run the battery on the cumulative model."""
    trial = AdmissionState(
        names=(*state.names, contribution.name),
        likelihoods=(*state.likelihoods, *contribution.likelihoods),
        parameters=(*state.parameters, *contribution.parameters),
        priors={**dict(state.priors), **dict(contribution.priors)},
        annotations=state.annotations,
    )
    spec, registry = _compile_partial(trial, causal_spec)
    bundle = build_prior_runtime_bundle(spec, registry)
    pred = sample_prior_predictive_from_runtime(
        spec, bundle, design.t_grid, num_samples=design.n_draws, seed=design.seed
    )

    latent_names, manifest_names, manifest_links = _spec_names(spec)
    d = latent_names.index(contribution.name)
    x = np.asarray(pred["latents"][:, :, d])
    dt = float(design.t_grid[1] - design.t_grid[0])

    results: list[CheckResult] = list(check_confinement(contribution.name, x, dt))

    # C2 scale anchor + C5 coverage come from the construct's indicator(s).
    anchor, anchor_src, anchor_detail = _scale_anchor(
        manifest_names, manifest_links, contribution, design.data
    )
    results.append(check_scale(contribution.name, x, anchor, anchor_src, anchor_detail))

    # C3 resolvability (only for dynamic constructs that own a potential well).
    comp_idx = _node_potential_index(spec, d)
    if comp_idx is not None:
        decay_site = f"vf_{comp_idx}_decay"
        if decay_site in pred:
            tau = 1.0 / np.asarray(pred[decay_site])
            results.append(check_resolvability(contribution.name, tau, design.cadence, design.span))

    # C4b edge overwhelm (edge-off re-simulation holds all else fixed).
    if contribution.edge_parents:
        edge_sites = _incoming_edge_sites(spec, contribution, latent_names, d)
        x_off = _resimulate_edge_off(spec, pred, design.t_grid, edge_sites, design.seed)[:, :, d]
        results.extend(
            check_edge_share(
                contribution.name, x[:, design.obs_idx], np.asarray(x_off)[:, design.obs_idx]
            )
        )

    # C4c Hill saturation (per saturating parent).
    for parent in contribution.hill_parents:
        p_idx = latent_names.index(parent)
        ec50_comp = _hill_ec50_index(spec, p_idx, d)
        ec50_site = f"vf_{ec50_comp}_EC50"
        if ec50_comp is not None and ec50_site in pred:
            parent_vals = np.asarray(pred["latents"][:, design.obs_idx, p_idx])
            results.append(
                check_saturation(
                    f"{parent}->{contribution.name}",
                    np.asarray(pred[ec50_site]),
                    parent_vals,
                )
            )

    # C5a/C5b/C5c coverage per indicator of this construct.
    for lik in contribution.likelihoods:
        m = manifest_names.index(lik.variable)
        lp = np.asarray(pred["linear_predictors"][:, design.obs_idx, m])
        pp_y = np.asarray(pred["observations"][:, design.obs_idx, m])
        signal = _signal_from_linear_predictor(manifest_links[m], lp)
        results.extend(check_coverage(lik.variable, pp_y, signal, design.data[lik.variable]))

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
) -> jnp.ndarray:
    """Re-simulate the latents with the given edge sites zeroed, all else fixed.

    Reuses the exact param draws (and Brownian path via ``seed``) from ``pred`` so
    the only difference from the edge-on trajectory is the zeroed edge — the
    same-noise contrast :func:`reachability.check_edge_share` expects.
    """
    from nof1_causal_lab.models.ssm.predictive.registry_runtime import (
        _simulate_vector_field_predictive_latents,
    )

    samples = dict(pred)
    for site in edge_sites:
        if site in samples:
            samples[site] = jnp.zeros_like(jnp.asarray(samples[site]))
    latents, _linear_predictors = _simulate_vector_field_predictive_latents(
        spec, samples, t_grid, transition_inputs=None, seed=seed
    )
    return latents


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #


def run_construct_build(
    causal_spec: dict,
    contributions: Mapping[str, ConstructContribution],
    design: DesignInfo,
    accepted: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[AdmissionState, list[AdmissionReport]]:
    """Admit every construct in topological order; stop at the first non-admit.

    Returns the accumulated :class:`AdmissionState` and the per-construct reports.
    A construct that is BLOCKED (hard failure) or NEEDS DECISION (unaccepted soft
    failure) halts the build — the caller revises its contribution (or accepts the
    consequence via ``accepted``) and re-runs. The final ``AdmissionState`` yields
    the ModelSpec + priors to compile once every construct is admitted.
    """
    accepted = accepted or {}
    order = build_construct_order(causal_spec)
    state = AdmissionState()
    reports: list[AdmissionReport] = []
    for name in order:
        state, report = admit_construct(
            state, contributions[name], causal_spec, design, accepted.get(name)
        )
        reports.append(report)
        if not report.admitted:
            break
    return state, reports
