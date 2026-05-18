"""Validate end-to-end artifact lineage across a pipeline run.

Reports cross-stage inconsistencies that no single stage's contract can catch
alone: schema drift between persisted payloads and their current contracts,
Stage 0 column descriptions disagreeing with the raw input parquet,
construct/indicator/outcome divergence across stages, Stage 1b inventing
causal edges not present in Stage 1a, outcome constructs with no observed
indicator, Stage 4 likelihoods or priors targeting variables/parameters that
don't exist, Stage 5b posteriors that disagree with the Stage 4 parameter
set, Stage 6 interventions on unknown or non-identifiable constructs, and
Stage 6 manifest effects keyed on unknown indicators.

Usage::

    uv run python scripts/validate_run.py --workspace-id DEMO
    uv run python scripts/validate_run.py --workspace-id DEMO --up-to stage-5b
    uv run python scripts/validate_run.py --workspace-id DEMO --strict

Exits 1 if any errors are found (or any warnings with ``--strict``), else 0.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import ValidationError

if TYPE_CHECKING:
    from collections.abc import Callable

from nof1_causal_lab.flows.run_store import (
    STAGE0_PARQUET_FILENAMES,
    STAGE2_MODEL_PARQUET_FILENAMES,
    find_run_artifact,
    load_parquet,
    load_public_payload,
)
from nof1_causal_lab.flows.stage_contracts import STAGE_CONTRACTS
from nof1_causal_lab.flows.stage_registry import get_execution_order
from nof1_causal_lab.utils import storage
from nof1_causal_lab.utils.data import runs_dir

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class LineageIssue:
    rule: str
    severity: Severity
    stages: tuple[str, ...]
    message: str


@dataclass(frozen=True)
class RunContext:
    workspace_id: str
    stages: dict[str, dict[str, Any]]
    stage_paths: dict[str, str]
    model_indicators: set[str] | None
    raw_input_columns: set[str] | None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _stage_json_path(workspace_id: str, stage_id: str) -> str:
    return storage.join(runs_dir(workspace_id), f"{stage_id}.json")


def load_run_context(workspace_id: str, *, up_to: str | None) -> RunContext:
    order = list(get_execution_order())
    if up_to is not None and up_to not in order:
        raise ValueError(f"Unknown stage '{up_to}'. Expected one of: {', '.join(order)}")
    if up_to is not None:
        order = order[: order.index(up_to) + 1]

    stages: dict[str, dict[str, Any]] = {}
    stage_paths: dict[str, str] = {}
    for stage_id in order:
        path = _stage_json_path(workspace_id, stage_id)
        if not storage.exists(path):
            continue
        stages[stage_id] = load_public_payload(workspace_id, stage_id)
        stage_paths[stage_id] = path

    model_indicators: set[str] | None = None
    if "stage-2" in stages:
        try:
            parquet_path = find_run_artifact(workspace_id, STAGE2_MODEL_PARQUET_FILENAMES)
            df = load_parquet(parquet_path)
            if "indicator" in df.columns:
                model_indicators = set(df["indicator"].unique().to_list())
        except FileNotFoundError:
            pass

    raw_input_columns: set[str] | None = None
    if "stage-0" in stages:
        try:
            raw_path = find_run_artifact(workspace_id, STAGE0_PARQUET_FILENAMES)
            raw_input_columns = set(load_parquet(raw_path).columns)
        except FileNotFoundError:
            pass

    return RunContext(
        workspace_id=workspace_id,
        stages=stages,
        stage_paths=stage_paths,
        model_indicators=model_indicators,
        raw_input_columns=raw_input_columns,
    )


# ---------------------------------------------------------------------------
# Payload helpers
# ---------------------------------------------------------------------------


def _construct_names(latent: dict[str, Any]) -> list[str]:
    return [c["name"] for c in latent.get("constructs", []) if isinstance(c, dict) and "name" in c]


def _construct_map(latent: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        c["name"]: c for c in latent.get("constructs", []) if isinstance(c, dict) and "name" in c
    }


def _outcome_name(latent: dict[str, Any]) -> str | None:
    for construct in latent.get("constructs", []):
        if isinstance(construct, dict) and construct.get("is_outcome"):
            return construct.get("name")
    return None


def _stage1b_indicators(stage_1b: dict[str, Any]) -> list[dict[str, Any]]:
    indicators = stage_1b.get("causal_spec", {}).get("measurement", {}).get("indicators", [])
    return [i for i in indicators if isinstance(i, dict)]


def _stage1b_indicator_names(stage_1b: dict[str, Any]) -> set[str]:
    return {i["name"] for i in _stage1b_indicators(stage_1b) if "name" in i}


# ---------------------------------------------------------------------------
# Rules — each returns a list of LineageIssue
# ---------------------------------------------------------------------------


def rule_contract_conformance(ctx: RunContext) -> list[LineageIssue]:
    issues: list[LineageIssue] = []
    for stage_id, payload in ctx.stages.items():
        contract = STAGE_CONTRACTS[stage_id]
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
                    stages=(stage_id,),
                    message=f"{stage_id}.json does not conform to {contract.__name__}: {head}{tail}",
                )
            )
    return issues


def rule_stage0_columns_match_raw_parquet(ctx: RunContext) -> list[LineageIssue]:
    if "stage-0" not in ctx.stages or ctx.raw_input_columns is None:
        return []
    described = {
        c["name"]
        for c in (ctx.stages["stage-0"].get("column_descriptions") or [])
        if isinstance(c, dict) and "name" in c
    }
    if not described:
        return []
    issues: list[LineageIssue] = []
    if extra := described - ctx.raw_input_columns:
        issues.append(
            LineageIssue(
                rule="stage0-columns-match-raw-parquet",
                severity="error",
                stages=("stage-0",),
                message=(
                    "stage-0 column_descriptions name columns absent from "
                    f"stage0-raw-input.parquet: {sorted(extra)}"
                ),
            )
        )
    if missing := ctx.raw_input_columns - described:
        issues.append(
            LineageIssue(
                rule="stage0-columns-match-raw-parquet",
                severity="error",
                stages=("stage-0",),
                message=(
                    "stage0-raw-input.parquet contains columns without a "
                    f"stage-0 column_descriptions entry: {sorted(missing)}"
                ),
            )
        )
    return issues


def rule_constructs_stable_1a_1b(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1a" not in ctx.stages or "stage-1b" not in ctx.stages:
        return []
    names_1a = set(_construct_names(ctx.stages["stage-1a"].get("latent_model", {})))
    names_1b = set(
        _construct_names(ctx.stages["stage-1b"].get("causal_spec", {}).get("latent", {}))
    )
    issues: list[LineageIssue] = []
    if only_1a := names_1a - names_1b:
        issues.append(
            LineageIssue(
                rule="constructs-stable-1a-1b",
                severity="error",
                stages=("stage-1a", "stage-1b"),
                message=f"Constructs in stage-1a but missing from stage-1b: {sorted(only_1a)}",
            )
        )
    if only_1b := names_1b - names_1a:
        issues.append(
            LineageIssue(
                rule="constructs-stable-1a-1b",
                severity="error",
                stages=("stage-1a", "stage-1b"),
                message=f"Constructs in stage-1b but missing from stage-1a: {sorted(only_1b)}",
            )
        )
    return issues


def rule_construct_attributes_stable(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1a" not in ctx.stages or "stage-1b" not in ctx.stages:
        return []
    map_1a = _construct_map(ctx.stages["stage-1a"].get("latent_model", {}))
    map_1b = _construct_map(ctx.stages["stage-1b"].get("causal_spec", {}).get("latent", {}))
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
                    stages=("stage-1a", "stage-1b"),
                    message=f"Construct '{name}' attributes differ between stage-1a and stage-1b: {', '.join(diffs)}",
                )
            )
    return issues


def rule_outcome_stable(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1a" not in ctx.stages or "stage-1b" not in ctx.stages:
        return []
    outcome_1a = _outcome_name(ctx.stages["stage-1a"].get("latent_model", {}))
    outcome_1b = _outcome_name(ctx.stages["stage-1b"].get("causal_spec", {}).get("latent", {}))
    if outcome_1a is None:
        return [
            LineageIssue(
                rule="outcome-stable",
                severity="error",
                stages=("stage-1a",),
                message="No construct with is_outcome=true in stage-1a",
            )
        ]
    if outcome_1a != outcome_1b:
        return [
            LineageIssue(
                rule="outcome-stable",
                severity="error",
                stages=("stage-1a", "stage-1b"),
                message=(
                    f"Outcome construct differs: stage-1a='{outcome_1a}' vs stage-1b='{outcome_1b}'"
                ),
            )
        ]
    return []


def _edge_tuples(edges: list[dict[str, Any]]) -> set[tuple[str, str, bool]]:
    return {
        (e["cause"], e["effect"], bool(e.get("lagged", True)))
        for e in edges
        if isinstance(e, dict) and "cause" in e and "effect" in e
    }


def rule_stage1b_edges_monotonic_1a(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1a" not in ctx.stages or "stage-1b" not in ctx.stages:
        return []
    e1a = _edge_tuples(ctx.stages["stage-1a"].get("latent_model", {}).get("edges") or [])
    e1b = _edge_tuples(
        ctx.stages["stage-1b"].get("causal_spec", {}).get("latent", {}).get("edges") or []
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
            rule="stage1b-edges-monotonic-1a",
            severity="error",
            stages=("stage-1a", "stage-1b"),
            message=f"stage-1b introduces edges not in stage-1a: {detail}",
        )
    ]


def rule_outcome_has_indicator(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1a" not in ctx.stages or "stage-1b" not in ctx.stages:
        return []
    outcome = _outcome_name(ctx.stages["stage-1a"].get("latent_model", {}))
    if outcome is None:
        return []  # rule-outcome-stable owns this case
    indicators_for_outcome = [
        i.get("name")
        for i in _stage1b_indicators(ctx.stages["stage-1b"])
        if i.get("construct_name") == outcome
    ]
    if indicators_for_outcome:
        return []
    return [
        LineageIssue(
            rule="outcome-has-indicator",
            severity="error",
            stages=("stage-1a", "stage-1b"),
            message=(
                f"Outcome construct '{outcome}' has no stage-1b indicators "
                "with matching construct_name; the model has no observed signal "
                "for the outcome and cannot be fit"
            ),
        )
    ]


def rule_source_columns_in_stage0(ctx: RunContext) -> list[LineageIssue]:
    if ctx.raw_input_columns is None or "stage-1b" not in ctx.stages:
        return []
    unknown: dict[str, list[str]] = {}
    for indicator in _stage1b_indicators(ctx.stages["stage-1b"]):
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
            rule="source-columns-in-stage0",
            severity="error",
            stages=("stage-0", "stage-1b"),
            message=(
                "stage-1b indicators reference source_columns not in "
                f"stage0-raw-input.parquet: {detail}"
            ),
        )
    ]


def rule_indicators_audited_by_stage3(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1b" not in ctx.stages or "stage-3" not in ctx.stages:
        return []
    indicators_1b = _stage1b_indicator_names(ctx.stages["stage-1b"])
    audited = set(ctx.stages["stage-3"].get("indicators", {}).keys())
    missing = indicators_1b - audited
    if not missing:
        return []
    return [
        LineageIssue(
            rule="indicators-audited-by-stage3",
            severity="error",
            stages=("stage-1b", "stage-3"),
            message=f"Indicators in stage-1b not audited by stage-3: {sorted(missing)}",
        )
    ]


def rule_indicators_in_model_data(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1b" not in ctx.stages or "stage-2" not in ctx.stages or ctx.model_indicators is None:
        return []
    indicators_1b = _stage1b_indicator_names(ctx.stages["stage-1b"])
    missing = indicators_1b - ctx.model_indicators
    if not missing:
        return []
    return [
        LineageIssue(
            rule="indicators-in-model-data",
            severity="warning",
            stages=("stage-1b", "stage-2"),
            message=(
                "Indicators declared in stage-1b but absent from stage2-model-data.parquet "
                f"(no extracted observations): {sorted(missing)}"
            ),
        )
    ]


def rule_likelihood_variables_in_1b_indicators(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1b" not in ctx.stages or "stage-4" not in ctx.stages:
        return []
    indicators = _stage1b_indicator_names(ctx.stages["stage-1b"])
    likelihoods = ctx.stages["stage-4"].get("model_spec", {}).get("likelihoods") or []
    used = {lk.get("variable") for lk in likelihoods if isinstance(lk, dict) and lk.get("variable")}
    unknown = used - indicators
    if not unknown:
        return []
    return [
        LineageIssue(
            rule="likelihood-variables-in-1b-indicators",
            severity="error",
            stages=("stage-1b", "stage-4"),
            message=(
                "stage-4 likelihoods reference variables not in stage-1b indicators: "
                f"{sorted(unknown)}"
            ),
        )
    ]


def rule_outcome_indicators_have_likelihoods(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1a" not in ctx.stages or "stage-1b" not in ctx.stages or "stage-4" not in ctx.stages:
        return []
    outcome = _outcome_name(ctx.stages["stage-1a"].get("latent_model", {}))
    if outcome is None:
        return []
    outcome_indicators = {
        i["name"]
        for i in _stage1b_indicators(ctx.stages["stage-1b"])
        if i.get("construct_name") == outcome and "name" in i
    }
    if not outcome_indicators:
        return []  # rule-outcome-has-indicator owns this case
    likelihoods = ctx.stages["stage-4"].get("model_spec", {}).get("likelihoods") or []
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
            stages=("stage-1b", "stage-4"),
            message=(
                f"Outcome '{outcome}' has stage-1b indicators without stage-4 "
                f"likelihoods: {sorted(missing)} (outcome cannot be fit from these)"
            ),
        )
    ]


def rule_stage4_priors_target_params(ctx: RunContext) -> list[LineageIssue]:
    # Only ``authored_priors`` is constrained to ``model_spec.parameters``.
    # ``resolved_priors`` is intentionally a superset: the SSM compiler adds
    # implicit ``t0_mean_<latent>`` / ``t0_sd_<latent>`` rows for every latent
    # construct via ``_build_compiled_initial_state_priors``, regardless of
    # whether the parameter is tracked in ``model_spec.parameters``.
    if "stage-4" not in ctx.stages:
        return []
    payload = ctx.stages["stage-4"]
    params = payload.get("model_spec", {}).get("parameters", []) or []
    param_names = {p["name"] for p in params if isinstance(p, dict) and "name" in p}
    authored = set((payload.get("authored_priors") or {}).keys())
    unknown = authored - param_names
    if not unknown:
        return []
    return [
        LineageIssue(
            rule="stage4-priors-target-params",
            severity="error",
            stages=("stage-4",),
            message=f"authored_priors target unknown parameters: {sorted(unknown)}",
        )
    ]


def rule_stage5b_posterior_covers_stage4_params(ctx: RunContext) -> list[LineageIssue]:
    if "stage-4" not in ctx.stages or "stage-5b" not in ctx.stages:
        return []
    params = ctx.stages["stage-4"].get("model_spec", {}).get("parameters") or []
    param_names = {p["name"] for p in params if isinstance(p, dict) and "name" in p}
    marginals = ctx.stages["stage-5b"].get("posterior_marginals") or []
    posterior_names = {
        m.get("parameter") for m in marginals if isinstance(m, dict) and m.get("parameter")
    }
    if not param_names or not posterior_names:
        return []
    # Some inference methods (e.g. laplace_em) expose compiled tensor names like
    # ``drift[0]`` rather than the Stage 4 user-facing names. When no overlap
    # exists, the namespaces are simply different and a name-set comparison is
    # not meaningful. Only enforce coverage when at least one name matches.
    if not (param_names & posterior_names):
        return []
    missing = param_names - posterior_names
    if not missing:
        return []
    return [
        LineageIssue(
            rule="stage5b-posterior-covers-stage4-params",
            severity="error",
            stages=("stage-4", "stage-5b"),
            message=(
                f"stage-4 parameters missing from stage-5b posterior_marginals: {sorted(missing)}"
            ),
        )
    ]


def rule_stage5b_posterior_pairs_in_marginals(ctx: RunContext) -> list[LineageIssue]:
    if "stage-5b" not in ctx.stages:
        return []
    payload = ctx.stages["stage-5b"]
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
            rule="stage5b-posterior-pairs-in-marginals",
            severity="error",
            stages=("stage-5b",),
            message=(
                "stage-5b posterior_pairs reference parameters not in "
                f"posterior_marginals: {sorted(missing)}"
            ),
        )
    ]


def rule_stage6_treatments_known(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1a" not in ctx.stages or "stage-6" not in ctx.stages:
        return []
    constructs = set(_construct_names(ctx.stages["stage-1a"].get("latent_model", {})))
    treatments = {
        ir.get("treatment")
        for ir in (ctx.stages["stage-6"].get("intervention_results") or [])
        if isinstance(ir, dict) and ir.get("treatment")
    }
    unknown = treatments - constructs
    if not unknown:
        return []
    return [
        LineageIssue(
            rule="stage6-treatments-known",
            severity="error",
            stages=("stage-1a", "stage-6"),
            message=f"stage-6 intervention treatments not in stage-1a constructs: {sorted(unknown)}",
        )
    ]


def rule_stage6_treatments_identifiable(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1b" not in ctx.stages or "stage-6" not in ctx.stages:
        return []
    treatments = {
        ir.get("treatment")
        for ir in (ctx.stages["stage-6"].get("intervention_results") or [])
        if isinstance(ir, dict) and ir.get("treatment")
    }
    if not treatments:
        return []
    ident = ctx.stages["stage-1b"].get("causal_spec", {}).get("identifiability")
    if not isinstance(ident, dict):
        return [
            LineageIssue(
                rule="stage6-treatments-identifiable",
                severity="error",
                stages=("stage-1b", "stage-6"),
                message=(
                    "stage-6 has intervention results but stage-1b has no identifiability verdicts"
                ),
            )
        ]
    identifiable = set((ident.get("identifiable_treatments") or {}).keys())
    violations = treatments - identifiable
    if not violations:
        return []
    return [
        LineageIssue(
            rule="stage6-treatments-identifiable",
            severity="error",
            stages=("stage-1b", "stage-6"),
            message=(
                "stage-6 ran interventions on treatments not explicitly listed in "
                f"stage-1b identifiable_treatments: {sorted(violations)}"
            ),
        )
    ]


def rule_stage6_manifest_effects_on_1b_indicators(ctx: RunContext) -> list[LineageIssue]:
    if "stage-1b" not in ctx.stages or "stage-6" not in ctx.stages:
        return []
    indicators = _stage1b_indicator_names(ctx.stages["stage-1b"])
    if not indicators:
        return []
    unknown_by_treatment: dict[str, list[str]] = {}
    for ir in ctx.stages["stage-6"].get("intervention_results") or []:
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
            rule="stage6-manifest-effects-on-1b-indicators",
            severity="error",
            stages=("stage-1b", "stage-6"),
            message=(
                f"stage-6 manifest_effects reference keys not in stage-1b indicators: {detail}"
            ),
        )
    ]


RULES: list[Callable[[RunContext], list[LineageIssue]]] = [
    rule_contract_conformance,
    rule_stage0_columns_match_raw_parquet,
    rule_constructs_stable_1a_1b,
    rule_construct_attributes_stable,
    rule_outcome_stable,
    rule_stage1b_edges_monotonic_1a,
    rule_outcome_has_indicator,
    rule_source_columns_in_stage0,
    rule_indicators_audited_by_stage3,
    rule_indicators_in_model_data,
    rule_likelihood_variables_in_1b_indicators,
    rule_outcome_indicators_have_likelihoods,
    rule_stage4_priors_target_params,
    rule_stage5b_posterior_covers_stage4_params,
    rule_stage5b_posterior_pairs_in_marginals,
    rule_stage6_treatments_known,
    rule_stage6_treatments_identifiable,
    rule_stage6_manifest_effects_on_1b_indicators,
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate cross-stage artifact lineage of a pipeline run"
    )
    parser.add_argument(
        "--workspace-id",
        required=True,
        help="Workspace ID under data/ (e.g. DEMO, GOLDEN, SMALLGOLDEN)",
    )
    parser.add_argument(
        "--up-to",
        default=None,
        help="Validate only up to this stage (e.g. stage-5b). Default: all present stages.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors (non-zero exit on any issue).",
    )
    args = parser.parse_args(argv)

    run_dir = runs_dir(args.workspace_id)
    print(f"Validating run: workspace={args.workspace_id} ({run_dir})")

    try:
        ctx = load_run_context(args.workspace_id, up_to=args.up_to)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not ctx.stages:
        print("error: no stage-*.json files found in run directory", file=sys.stderr)
        return 1

    print(f"Stages found: {', '.join(ctx.stages)}")
    if ctx.model_indicators is not None:
        print(f"Model-data indicators: {len(ctx.model_indicators)} unique")
    else:
        print("Model-data indicators: (stage2-model-data.parquet not found)")

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
        scope = "+".join(issue.stages)
        print(f"  [{prefix}] {issue.rule} ({scope}): {issue.message}")

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
