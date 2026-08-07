"""Validate end-to-end artifact lineage across a pipeline run.

Reports cross-artifact inconsistencies that no single artifact's contract can catch
alone: schema drift between persisted payloads and their current contracts,
raw_data column descriptions disagreeing with the raw input parquet,
construct/indicator/outcome divergence across artifacts, measurement_structure inventing
causal edges not present in latent_structure, outcome constructs with no observed
indicator, statistical_model_spec likelihoods or priors targeting variables/parameters that
don't exist, posterior posteriors that disagree with the statistical_model_spec parameter
set, baseline_report interventions on unknown or non-identifiable constructs, and
baseline_report manifest effects keyed on unknown indicators.

Usage::

    uv run python scripts/validate_run.py --workspace-id DEMO
    uv run python scripts/validate_run.py --workspace-id DEMO --up-to posterior
    uv run python scripts/validate_run.py --workspace-id DEMO --strict

Exits 1 if any errors are found (or any warnings with ``--strict``), else 0.
"""

from __future__ import annotations

import argparse
import graphlib
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import ValidationError
from scripts.artifact_contract_catalog import ARTIFACT_CONTRACTS

if TYPE_CHECKING:
    from collections.abc import Callable

    from nof1_causal_lab.machine.artifacts import ArtifactId


from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001
from nof1_causal_lab.machine.artifact_files import json_filename, parquet_filename
from nof1_causal_lab.machine.graph import ARTIFACT_GRAPH, DERIVATIONS
from nof1_causal_lab.machine.store import ArtifactStore, derive_current_state
from nof1_causal_lab.utils import storage

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class LineageIssue:
    rule: str
    severity: Severity
    artifacts: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class RunContext:
    workspace_id: str
    artifacts: dict[str, UncheckedJsonObject]
    artifact_paths: dict[str, str]
    model_indicators: set[str] | None
    raw_input_columns: set[str] | None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

RESULT_ARTIFACTS: dict[ArtifactId, str] = {
    "raw_data": "profile",
    "latent_structure": "latent_structure",
    "measurement_structure": "measurement_structure",
    "causal_design": "causal_design",
    "measurements": "measurements",
    "validation_report": "validation_report",
    "statistical_model_spec": "statistical_model_spec",
    "posterior": "diagnostics",
    "baseline_report": "baseline_report",
}


def load_parquet(path: str) -> Any:
    """Read a parquet frame from local or remote workspace storage."""
    import polars as pl

    return pl.read_parquet(path, storage_options=storage.polars_storage_options())


def current_artifact_file(workspace_id: str, artifact_id: ArtifactId, filename: str) -> str:
    """Resolve a file in the current journal-derived artifact version."""
    info = derive_current_state(workspace_id).get(artifact_id)
    if info is None:
        raise FileNotFoundError(f"No current '{artifact_id}' artifact for workspace {workspace_id}")
    return ArtifactStore(workspace_id).file_path(artifact_id, info.version, filename)


def _result_artifact_order() -> tuple[ArtifactId, ...]:
    result_artifacts = set(RESULT_ARTIFACTS)
    dependencies: dict[ArtifactId, set[ArtifactId]] = {
        artifact_id: set() for artifact_id in result_artifacts
    }
    for spec in ARTIFACT_GRAPH:
        for artifact_id in spec.all_produces:
            if artifact_id in dependencies:
                dependencies[artifact_id].update(
                    parent for parent in spec.consumes if parent in result_artifacts
                )
    for spec in DERIVATIONS:
        if spec.produces in dependencies:
            dependencies[spec.produces].update(
                parent for parent in spec.from_ if parent in result_artifacts
            )
    return tuple(graphlib.TopologicalSorter(dependencies).static_order())


def _artifact_order() -> tuple[str, ...]:
    return tuple(_result_artifact_order())


def load_run_context(workspace_id: str, *, up_to: str | None) -> RunContext:
    artifact_ids = list(_artifact_order())
    if up_to is not None and up_to not in artifact_ids:
        raise ValueError(f"Unknown artifact '{up_to}'. Expected one of: {', '.join(artifact_ids)}")
    artifact_order = list(_result_artifact_order())
    if up_to is not None:
        artifact_order = artifact_order[: artifact_order.index(cast("ArtifactId", up_to)) + 1]

    artifacts: dict[str, UncheckedJsonObject] = {}
    artifact_paths: dict[str, str] = {}
    state = derive_current_state(workspace_id)
    store = ArtifactStore(workspace_id)
    for artifact_id in artifact_order:
        key = RESULT_ARTIFACTS[artifact_id]
        info = state.get(artifact_id)
        if info is None:
            continue
        filename = json_filename(artifact_id, key)
        payload = store.read_json_file(artifact_id, info.version, filename)
        if not isinstance(payload, dict):
            raise TypeError(
                f"Canonical payload for {artifact_id} ({artifact_id}/{filename}) is not a dict"
            )
        artifacts[artifact_id] = payload
        artifact_paths[artifact_id] = store.file_path(artifact_id, info.version, filename)

    model_indicators: set[str] | None = None
    if "measurements" in artifacts:
        try:
            parquet_path = current_artifact_file(
                workspace_id,
                "panel",
                parquet_filename("panel", "panel"),
            )
            df = load_parquet(parquet_path)
            if "indicator" in df.columns:
                model_indicators = set(df["indicator"].unique().to_list())
        except FileNotFoundError:
            pass

    raw_input_columns: set[str] | None = None
    if "raw_data" in artifacts:
        try:
            raw_path = current_artifact_file(
                workspace_id,
                "raw_data",
                parquet_filename("raw_data", "raw"),
            )
            raw_input_columns = set(load_parquet(raw_path).columns)
        except FileNotFoundError:
            pass

    return RunContext(
        workspace_id=workspace_id,
        artifacts=artifacts,
        artifact_paths=artifact_paths,
        model_indicators=model_indicators,
        raw_input_columns=raw_input_columns,
    )


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _construct_names(latent: UncheckedJsonObject) -> list[str]:
    return [c["name"] for c in latent.get("constructs", []) if isinstance(c, dict) and "name" in c]


def _construct_map(latent: UncheckedJsonObject) -> dict[str, UncheckedJsonObject]:
    return {
        c["name"]: c for c in latent.get("constructs", []) if isinstance(c, dict) and "name" in c
    }


def _outcome_name(latent: UncheckedJsonObject) -> str | None:
    for construct in latent.get("constructs", []):
        if isinstance(construct, dict) and construct.get("is_outcome"):
            return construct.get("name")
    return None


def _causal_design_indicators(
    causal_design: UncheckedJsonObject,
) -> list[UncheckedJsonObject]:
    indicators = causal_design.get("causal_design", {}).get("measurement", {}).get("indicators", [])
    return [i for i in indicators if isinstance(i, dict)]


def _causal_design_indicator_names(causal_design: UncheckedJsonObject) -> set[str]:
    return {i["name"] for i in _causal_design_indicators(causal_design) if "name" in i}


# ---------------------------------------------------------------------------
# Rules — each returns a list of LineageIssue
# ---------------------------------------------------------------------------


def rule_contract_conformance(ctx: RunContext) -> list[LineageIssue]:
    issues: list[LineageIssue] = []
    for artifact_id, payload in ctx.artifacts.items():
        contract = ARTIFACT_CONTRACTS.get(artifact_id)
        if contract is None:
            continue
        try:
            contract.model_validate(payload)
        except ValidationError as exc:
            errs = exc.errors()
            head = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in errs[:3])
            tail = f" (+{len(errs) - 3} more)" if len(errs) > 3 else ""
            issues.append(
                LineageIssue(
                    rule="contract-conformance",
                    severity="error",
                    artifacts=(artifact_id,),
                    message=f"{artifact_id}.json does not conform to {contract.__name__}: {head}{tail}",
                )
            )
    return issues


def rule_raw_data_columns_match_raw_parquet(ctx: RunContext) -> list[LineageIssue]:
    if "raw_data" not in ctx.artifacts or ctx.raw_input_columns is None:
        return []
    described = {
        c["name"]
        for c in (ctx.artifacts["raw_data"].get("column_descriptions") or [])
        if isinstance(c, dict) and "name" in c
    }
    if not described:
        return []
    issues: list[LineageIssue] = []
    if extra := described - ctx.raw_input_columns:
        issues.append(
            LineageIssue(
                rule="raw-data-columns-match-raw-parquet",
                severity="error",
                artifacts=("raw_data",),
                message=(
                    "raw_data column_descriptions name columns absent from "
                    f"raw_data/raw.parquet: {sorted(extra)}"
                ),
            )
        )
    if missing := ctx.raw_input_columns - described:
        issues.append(
            LineageIssue(
                rule="raw-data-columns-match-raw-parquet",
                severity="error",
                artifacts=("raw_data",),
                message=(
                    "raw_data/raw.parquet contains columns without a "
                    f"raw_data column_descriptions entry: {sorted(missing)}"
                ),
            )
        )
    return issues


def rule_constructs_stable(ctx: RunContext) -> list[LineageIssue]:
    if "latent_structure" not in ctx.artifacts or "causal_design" not in ctx.artifacts:
        return []
    names_1a = set(_construct_names(ctx.artifacts["latent_structure"].get("latent_structure", {})))
    names_1b = set(
        _construct_names(ctx.artifacts["causal_design"].get("causal_design", {}).get("latent", {}))
    )
    issues: list[LineageIssue] = []
    if only_1a := names_1a - names_1b:
        issues.append(
            LineageIssue(
                rule="constructs-stable",
                severity="error",
                artifacts=("latent_structure", "causal_design"),
                message=f"Constructs in latent_structure but missing from causal_design: {sorted(only_1a)}",
            )
        )
    if only_1b := names_1b - names_1a:
        issues.append(
            LineageIssue(
                rule="constructs-stable",
                severity="error",
                artifacts=("latent_structure", "causal_design"),
                message=f"Constructs in causal_design but missing from latent_structure: {sorted(only_1b)}",
            )
        )
    return issues


def rule_construct_attributes_stable(ctx: RunContext) -> list[LineageIssue]:
    if "latent_structure" not in ctx.artifacts or "causal_design" not in ctx.artifacts:
        return []
    map_1a = _construct_map(ctx.artifacts["latent_structure"].get("latent_structure", {}))
    map_1b = _construct_map(
        ctx.artifacts["causal_design"].get("causal_design", {}).get("latent", {})
    )
    attrs = ("role", "temporal_status", "is_outcome")
    issues: list[LineageIssue] = []
    for name in sorted(map_1a.keys() & map_1b.keys()):
        c1a, c1b = map_1a[name], map_1b[name]
        diffs = [f"{a}={c1a.get(a)!r} vs {c1b.get(a)!r}" for a in attrs if c1a.get(a) != c1b.get(a)]
        if diffs:
            issues.append(
                LineageIssue(
                    rule="construct-attributes-stable",
                    severity="error",
                    artifacts=("latent_structure", "causal_design"),
                    message=f"Construct '{name}' attributes differ between latent_structure and causal_design: {', '.join(diffs)}",
                )
            )
    return issues


def rule_outcome_stable(ctx: RunContext) -> list[LineageIssue]:
    if "latent_structure" not in ctx.artifacts or "causal_design" not in ctx.artifacts:
        return []
    outcome_1a = _outcome_name(ctx.artifacts["latent_structure"].get("latent_structure", {}))
    outcome_1b = _outcome_name(
        ctx.artifacts["causal_design"].get("causal_design", {}).get("latent", {})
    )
    if outcome_1a is None:
        return [
            LineageIssue(
                rule="outcome-stable",
                severity="error",
                artifacts=("latent_structure",),
                message="No construct with is_outcome=true in latent_structure",
            )
        ]
    if outcome_1a != outcome_1b:
        return [
            LineageIssue(
                rule="outcome-stable",
                severity="error",
                artifacts=("latent_structure", "causal_design"),
                message=(
                    f"Outcome construct differs: latent_structure='{outcome_1a}' vs causal_design='{outcome_1b}'"
                ),
            )
        ]
    return []


def _edge_tuples(edges: list[UncheckedJsonObject]) -> set[tuple[str, str, bool]]:
    return {
        (e["cause"], e["effect"], bool(e.get("lagged", True)))
        for e in edges
        if isinstance(e, dict) and "cause" in e and "effect" in e
    }


def rule_causal_design_edges_monotonic(ctx: RunContext) -> list[LineageIssue]:
    if "latent_structure" not in ctx.artifacts or "causal_design" not in ctx.artifacts:
        return []
    e1a = _edge_tuples(
        ctx.artifacts["latent_structure"].get("latent_structure", {}).get("edges") or []
    )
    e1b = _edge_tuples(
        ctx.artifacts["causal_design"].get("causal_design", {}).get("latent", {}).get("edges") or []
    )
    invented = e1b - e1a
    if not invented:
        return []
    detail = ", ".join(
        f"{cause}->{effect} ({'lagged' if lagged else 'contemporaneous'})"
        for cause, effect, lagged in sorted(invented)
    )
    return [
        LineageIssue(
            rule="causal-design-edges-monotonic",
            severity="error",
            artifacts=("latent_structure", "causal_design"),
            message=f"causal_design introduces edges not in latent_structure: {detail}",
        )
    ]


def rule_outcome_has_indicator(ctx: RunContext) -> list[LineageIssue]:
    if "latent_structure" not in ctx.artifacts or "causal_design" not in ctx.artifacts:
        return []
    outcome = _outcome_name(ctx.artifacts["latent_structure"].get("latent_structure", {}))
    if outcome is None:
        return []  # rule-outcome-stable owns this case
    indicators_for_outcome = [
        i.get("name")
        for i in _causal_design_indicators(ctx.artifacts["causal_design"])
        if i.get("construct_name") == outcome
    ]
    if indicators_for_outcome:
        return []
    return [
        LineageIssue(
            rule="outcome-has-indicator",
            severity="error",
            artifacts=("latent_structure", "causal_design"),
            message=(
                f"Outcome construct '{outcome}' has no causal_design indicators "
                "with matching construct_name; the model has no observed signal "
                "for the outcome and cannot be fit"
            ),
        )
    ]


def rule_source_columns_in_raw_data(ctx: RunContext) -> list[LineageIssue]:
    if ctx.raw_input_columns is None or "causal_design" not in ctx.artifacts:
        return []
    unknown: dict[str, list[str]] = {}
    for indicator in _causal_design_indicators(ctx.artifacts["causal_design"]):
        bad = [
            source_column
            for source_column in (indicator.get("source_columns") or [])
            if source_column not in ctx.raw_input_columns
        ]
        if bad:
            unknown[indicator.get("name", "?")] = sorted(bad)
    if not unknown:
        return []
    detail = "; ".join(f"{name}: {cols}" for name, cols in sorted(unknown.items()))
    return [
        LineageIssue(
            rule="source-columns-in-raw-data",
            severity="error",
            artifacts=("raw_data", "causal_design"),
            message=(
                "causal_design indicators reference source_columns not in "
                f"raw_data/raw.parquet: {detail}"
            ),
        )
    ]


def rule_indicators_audited_by_validation_report(ctx: RunContext) -> list[LineageIssue]:
    if "causal_design" not in ctx.artifacts or "validation_report" not in ctx.artifacts:
        return []
    indicators_1b = _causal_design_indicator_names(ctx.artifacts["causal_design"])
    audited = set(ctx.artifacts["validation_report"].get("indicators", {}).keys())
    missing = indicators_1b - audited
    if not missing:
        return []
    return [
        LineageIssue(
            rule="indicators-audited-by-validation-report",
            severity="error",
            artifacts=("causal_design", "validation_report"),
            message=f"Indicators in causal_design not audited by validation_report: {sorted(missing)}",
        )
    ]


def rule_indicators_in_panel(ctx: RunContext) -> list[LineageIssue]:
    if (
        "causal_design" not in ctx.artifacts
        or "measurements" not in ctx.artifacts
        or ctx.model_indicators is None
    ):
        return []
    indicators_1b = _causal_design_indicator_names(ctx.artifacts["causal_design"])
    missing = indicators_1b - ctx.model_indicators
    if not missing:
        return []
    return [
        LineageIssue(
            rule="indicators-in-panel",
            severity="warning",
            artifacts=("causal_design", "measurements"),
            message=(
                "Indicators declared in causal_design but absent from panel/panel.parquet "
                f"(no extracted observations): {sorted(missing)}"
            ),
        )
    ]


def rule_likelihood_variables_in_causal_design_indicators(ctx: RunContext) -> list[LineageIssue]:
    if "causal_design" not in ctx.artifacts or "statistical_model_spec" not in ctx.artifacts:
        return []
    indicators = _causal_design_indicator_names(ctx.artifacts["causal_design"])
    likelihoods = (
        ctx.artifacts["statistical_model_spec"].get("statistical_model_spec", {}).get("likelihoods")
        or []
    )
    used = {lk.get("variable") for lk in likelihoods if isinstance(lk, dict) and lk.get("variable")}
    unknown = used - indicators
    if not unknown:
        return []
    return [
        LineageIssue(
            rule="likelihood-variables-in-causal-design-indicators",
            severity="error",
            artifacts=("causal_design", "statistical_model_spec"),
            message=(
                "statistical_model_spec likelihoods reference variables not in causal_design indicators: "
                f"{sorted(unknown)}"
            ),
        )
    ]


def rule_outcome_indicators_have_likelihoods(ctx: RunContext) -> list[LineageIssue]:
    if (
        "latent_structure" not in ctx.artifacts
        or "causal_design" not in ctx.artifacts
        or "statistical_model_spec" not in ctx.artifacts
    ):
        return []
    outcome = _outcome_name(ctx.artifacts["latent_structure"].get("latent_structure", {}))
    if outcome is None:
        return []
    outcome_indicators = {
        i["name"]
        for i in _causal_design_indicators(ctx.artifacts["causal_design"])
        if i.get("construct_name") == outcome and "name" in i
    }
    if not outcome_indicators:
        return []  # rule-outcome-has-indicator owns this case
    likelihoods = (
        ctx.artifacts["statistical_model_spec"].get("statistical_model_spec", {}).get("likelihoods")
        or []
    )
    likelihood_vars = {
        lk.get("variable") for lk in likelihoods if isinstance(lk, dict) and lk.get("variable")
    }
    missing = outcome_indicators - likelihood_vars
    if not missing:
        return []
    return [
        LineageIssue(
            rule="outcome-indicators-have-likelihoods",
            severity="error",
            artifacts=("causal_design", "statistical_model_spec"),
            message=(
                f"Outcome '{outcome}' has causal_design indicators without statistical_model_spec "
                f"likelihoods: {sorted(missing)} (outcome cannot be fit from these)"
            ),
        )
    ]


def rule_statistical_model_spec_priors_target_params(ctx: RunContext) -> list[LineageIssue]:
    # Only ``authored_priors`` is constrained to ``statistical_model_spec.parameters``.
    # ``resolved_priors`` is intentionally a superset: the SSM compiler adds
    # implicit ``t0_mean_<latent>`` / ``t0_sd_<latent>`` rows for every latent
    # construct via ``_build_compiled_initial_state_priors``, regardless of
    # whether the parameter is tracked in ``statistical_model_spec.parameters``.
    if "statistical_model_spec" not in ctx.artifacts:
        return []
    payload = ctx.artifacts["statistical_model_spec"]
    params = payload.get("statistical_model_spec", {}).get("parameters", []) or []
    param_names = {p["name"] for p in params if isinstance(p, dict) and "name" in p}
    authored = set((payload.get("authored_priors") or {}).keys())
    unknown = authored - param_names
    if not unknown:
        return []
    return [
        LineageIssue(
            rule="statistical-model-spec-priors-target-params",
            severity="error",
            artifacts=("statistical_model_spec",),
            message=f"authored_priors target unknown parameters: {sorted(unknown)}",
        )
    ]


def rule_posterior_covers_statistical_model_spec_params(ctx: RunContext) -> list[LineageIssue]:
    if "statistical_model_spec" not in ctx.artifacts or "posterior" not in ctx.artifacts:
        return []
    params = (
        ctx.artifacts["statistical_model_spec"].get("statistical_model_spec", {}).get("parameters")
        or []
    )
    param_names = {p["name"] for p in params if isinstance(p, dict) and "name" in p}
    marginals = ctx.artifacts["posterior"].get("posterior_marginals") or []
    posterior_names = {
        m.get("parameter") for m in marginals if isinstance(m, dict) and m.get("parameter")
    }
    if not param_names or not posterior_names:
        return []
    # Some inference methods (e.g. laplace_em) expose compiled tensor names like
    # ``drift[0]`` rather than the statistical_model_spec user-facing names. When no overlap
    # exists, the namespaces are simply different and a name-set comparison is
    # not meaningful. Only enforce coverage when at least one name matches.
    if not (param_names & posterior_names):
        return []
    missing = param_names - posterior_names
    if not missing:
        return []
    return [
        LineageIssue(
            rule="posterior-covers-statistical-model-spec-params",
            severity="error",
            artifacts=("statistical_model_spec", "posterior"),
            message=(
                f"statistical_model_spec parameters missing from posterior_marginals: {sorted(missing)}"
            ),
        )
    ]


def rule_posterior_pairs_in_marginals(ctx: RunContext) -> list[LineageIssue]:
    if "posterior" not in ctx.artifacts:
        return []
    payload = ctx.artifacts["posterior"]
    marginals = payload.get("posterior_marginals") or []
    marginal_names = {
        m.get("parameter") for m in marginals if isinstance(m, dict) and m.get("parameter")
    }
    pairs = payload.get("posterior_pairs") or []
    if not marginal_names or not pairs:
        return []
    referenced = set()
    for pair in pairs:
        if not isinstance(pair, dict):
            continue
        for key in ("param_x", "param_y"):
            value = pair.get(key)
            if isinstance(value, str):
                referenced.add(value)
    missing = referenced - marginal_names
    if not missing:
        return []
    return [
        LineageIssue(
            rule="posterior-pairs-in-marginals",
            severity="error",
            artifacts=("posterior",),
            message=(
                "posterior_pairs reference parameters not in "
                f"posterior_marginals: {sorted(missing)}"
            ),
        )
    ]


def rule_baseline_report_treatments_known(ctx: RunContext) -> list[LineageIssue]:
    if "latent_structure" not in ctx.artifacts or "baseline_report" not in ctx.artifacts:
        return []
    constructs = set(
        _construct_names(ctx.artifacts["latent_structure"].get("latent_structure", {}))
    )
    treatments = {
        ir.get("treatment")
        for ir in (ctx.artifacts["baseline_report"].get("intervention_results") or [])
        if isinstance(ir, dict) and ir.get("treatment")
    }
    unknown = treatments - constructs
    if not unknown:
        return []
    return [
        LineageIssue(
            rule="baseline-report-treatments-known",
            severity="error",
            artifacts=("latent_structure", "baseline_report"),
            message=f"baseline_report intervention treatments not in latent_structure constructs: {sorted(unknown)}",
        )
    ]


def rule_baseline_report_treatments_identifiable(ctx: RunContext) -> list[LineageIssue]:
    if "causal_design" not in ctx.artifacts or "baseline_report" not in ctx.artifacts:
        return []
    treatments = {
        ir.get("treatment")
        for ir in (ctx.artifacts["baseline_report"].get("intervention_results") or [])
        if isinstance(ir, dict) and ir.get("treatment")
    }
    if not treatments:
        return []
    ident = ctx.artifacts["causal_design"].get("causal_design", {}).get("identifiability")
    if not isinstance(ident, dict):
        return [
            LineageIssue(
                rule="baseline-report-treatments-identifiable",
                severity="error",
                artifacts=("causal_design", "baseline_report"),
                message=(
                    "baseline_report has intervention results but causal_design has no identifiability verdicts"
                ),
            )
        ]
    identifiable = set((ident.get("identifiable_treatments") or {}).keys())
    violations = treatments - identifiable
    if not violations:
        return []
    return [
        LineageIssue(
            rule="baseline-report-treatments-identifiable",
            severity="error",
            artifacts=("causal_design", "baseline_report"),
            message=(
                "baseline_report ran interventions on treatments not explicitly listed in "
                f"causal_design identifiable_treatments: {sorted(violations)}"
            ),
        )
    ]


def rule_baseline_report_manifest_effects_on_causal_design_indicators(
    ctx: RunContext,
) -> list[LineageIssue]:
    if "causal_design" not in ctx.artifacts or "baseline_report" not in ctx.artifacts:
        return []
    indicators = _causal_design_indicator_names(ctx.artifacts["causal_design"])
    if not indicators:
        return []
    unknown_by_treatment: dict[str, list[str]] = {}
    for ir in ctx.artifacts["baseline_report"].get("intervention_results") or []:
        if not isinstance(ir, dict):
            continue
        effects = ir.get("manifest_effects") or {}
        if not isinstance(effects, dict):
            continue
        bad = [k for k in effects if k not in indicators]
        if bad:
            unknown_by_treatment[ir.get("treatment", "?")] = sorted(bad)
    if not unknown_by_treatment:
        return []
    detail = "; ".join(
        f"{treatment}: {keys}" for treatment, keys in sorted(unknown_by_treatment.items())
    )
    return [
        LineageIssue(
            rule="baseline-report-manifest-effects-on-causal-design-indicators",
            severity="error",
            artifacts=("causal_design", "baseline_report"),
            message=(
                f"baseline_report manifest_effects reference keys not in causal_design indicators: {detail}"
            ),
        )
    ]


RULES: list[Callable[[RunContext], list[LineageIssue]]] = [
    rule_contract_conformance,
    rule_raw_data_columns_match_raw_parquet,
    rule_constructs_stable,
    rule_construct_attributes_stable,
    rule_outcome_stable,
    rule_causal_design_edges_monotonic,
    rule_outcome_has_indicator,
    rule_source_columns_in_raw_data,
    rule_indicators_audited_by_validation_report,
    rule_indicators_in_panel,
    rule_likelihood_variables_in_causal_design_indicators,
    rule_outcome_indicators_have_likelihoods,
    rule_statistical_model_spec_priors_target_params,
    rule_posterior_covers_statistical_model_spec_params,
    rule_posterior_pairs_in_marginals,
    rule_baseline_report_treatments_known,
    rule_baseline_report_treatments_identifiable,
    rule_baseline_report_manifest_effects_on_causal_design_indicators,
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate cross-artifact artifact lineage of a pipeline run"
    )
    parser.add_argument(
        "--workspace-id",
        required=True,
        help="Workspace ID under data/ (e.g. DEMO)",
    )
    parser.add_argument(
        "--up-to",
        default=None,
        help="Validate only up to this artifact (e.g. posterior). Default: all present artifacts.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit on any issue).",
    )
    args = parser.parse_args(argv)

    print(f"Validating run: workspace={args.workspace_id}")

    try:
        ctx = load_run_context(args.workspace_id, up_to=args.up_to)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not ctx.artifacts:
        print("error: no current artifact result artifacts found in episode state", file=sys.stderr)
        return 1

    print(f"Artifacts found: {', '.join(ctx.artifacts)}")
    if ctx.model_indicators is not None:
        print(f"Panel indicators: {len(ctx.model_indicators)} unique")
    else:
        print("Panel indicators: (panel/panel.parquet not found)")

    issues: list[LineageIssue] = []
    for rule in RULES:
        issues.extend(rule(ctx))

    if not issues:
        print("\nLineage: OK")
        return 0

    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    print(f"\nFound {len(errors)} error(s) and {len(warnings)} warning(s):")
    for issue in issues:
        prefix = "ERROR" if issue.severity == "error" else "WARN "
        scope = "+".join(issue.artifacts)
        print(f"  [{prefix}] {issue.rule} ({scope}): {issue.message}")

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
