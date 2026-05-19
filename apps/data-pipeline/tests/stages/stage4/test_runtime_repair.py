"""Stage 4 runtime projection and repair routing tests."""

from tests.stages.stage4._support import (
    AssemblyValidation,
    PriorValidationResult,
    Stage4FrontierBlock,
    Stage4RepairCampaignState,
    _make_plan,
    _make_stage4_mechanics_context,
    _require_plan_block,
    _set_runtime_block,
    _with_positive_indicator_polarity,
    build_stage4_plan,
    classify_prior_failure_blocks,
    derive_deterministic_spec,
    make_stage4_runtime,
    project_stage4_graph,
    project_stage4_initial_state,
    project_stage4_snapshot,
    pytest,
)


def test_project_stage4_graph_includes_repair_barrier_and_prior_review_route():
    plan = _make_plan(
        model_blocks=(
            Stage4FrontierBlock(
                id="indicator:sleep_quality",
                kind="indicator_decision",
                label="Sleep Quality",
                variable_names=("sleep_quality",),
            ),
        ),
        review_block=Stage4FrontierBlock(
            id="review:model_spec",
            kind="global_review",
            label="Review model spec",
        ),
        prior_blocks=(
            Stage4FrontierBlock(
                id="effects:sleep",
                kind="effect_prior",
                label="Sleep effects",
                parameter_names=("beta_stress_sleep",),
            ),
        ),
        prior_review_block=Stage4FrontierBlock(
            id="review:prior_system",
            kind="global_prior_review",
            label="Review prior system",
        ),
    )

    graph = project_stage4_graph(plan)
    node_ids = {node["id"] for node in graph["nodes"]}
    edge_pairs = {(edge["from"], edge["to"], edge["kind"]) for edge in graph["edges"]}

    assert "__repair_barrier__" in node_ids
    assert "review:prior_system" in node_ids
    assert ("effects:sleep", "review:prior_system", "repair_transition") in edge_pairs
    assert ("__repair_barrier__", "review:prior_system", "repair_transition") in edge_pairs
    assert ("review:prior_system", "__done__", "phase_advance") in edge_pairs


def test_project_stage4_snapshot_serializes_waiting_block_cursor():
    plan = _make_plan(
        prior_blocks=(
            Stage4FrontierBlock(
                id="effects:sleep",
                kind="effect_prior",
                label="Sleep effects",
                parameter_names=("beta_stress_sleep",),
            ),
            Stage4FrontierBlock(
                id="effects:stress",
                kind="effect_prior",
                label="Stress effects",
                parameter_names=("beta_sleep_stress",),
            ),
        ),
    )
    runtime = make_stage4_runtime(plan)
    _set_runtime_block(plan, runtime, "effects:stress")

    snapshot = project_stage4_snapshot(plan, runtime)

    assert snapshot["cursor"] == {"kind": "block", "block_id": "effects:stress"}


def test_project_stage4_initial_state_matches_initial_plan_runtime():
    causal_spec, skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
    del skeleton, _data_for_model

    graph, snapshot = project_stage4_initial_state(causal_spec)

    assert graph == project_stage4_graph(plan)
    assert snapshot == project_stage4_snapshot(plan, runtime)


def test_scale_mismatch_for_single_indicator_construct_routes_to_dynamics_block():
    causal_spec = _with_positive_indicator_polarity(
        {
            "latent": {
                "constructs": [
                    {
                        "name": "chronotype",
                        "role": "exogenous",
                        "temporal_status": "time_invariant",
                    },
                    {
                        "name": "sleep_quality",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "is_outcome": True,
                    },
                ],
                "edges": [
                    {"cause": "chronotype", "effect": "sleep_quality", "lagged": False},
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "monthly_eveningness_activity_timing",
                        "construct_name": "chronotype",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                        "observation_window": "1mo",
                    },
                    {
                        "name": "sleep_quality_search_count",
                        "construct_name": "sleep_quality",
                        "measurement_dtype": "count",
                        "aggregation": "sum",
                    },
                ],
            },
            "estimation": {
                "state_order": ["chronotype", "sleep_quality"],
                "edges": [
                    {"cause": "chronotype", "effect": "sleep_quality", "lagged": False},
                ],
                "induced_dependencies": [],
            },
        }
    )
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    runtime = make_stage4_runtime(plan)
    active_block = _require_plan_block(plan, "dynamics:sleep_quality")

    repair_plan = classify_prior_failure_blocks(
        plan,
        active_block,
        AssemblyValidation(
            normalized_model_spec={
                "likelihoods": [
                    {
                        "variable": "monthly_eveningness_activity_timing",
                        "distribution": "gaussian",
                        "link": "identity",
                        "centered": True,
                    }
                ],
                "parameters": [
                    {
                        "name": "t0_mean_chronotype",
                        "role": "initial_state_mean",
                        "construct": "chronotype",
                    },
                    {
                        "name": "t0_sd_chronotype",
                        "role": "initial_state_sd",
                        "construct": "chronotype",
                    },
                    {
                        "name": "beta_chronotype_sleep_quality",
                        "role": "fixed_effect",
                        "cause": "chronotype",
                        "effect": "sleep_quality",
                    },
                ],
            },
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="scale_monthly_eveningness_activity_timing",
                    is_valid=False,
                    code="scale_mismatch",
                    origin="prior_predictive",
                    issue="Scale mismatch for monthly_eveningness_activity_timing",
                    suggested_adjustment="Adjust diffusion/drift priors to match data scale",
                )
            ],
        ),
        runtime,
    )

    assert repair_plan.scope.scope_kind == "direct_writer_blocks"
    assert repair_plan.scope.parameter_names == ("t0_sd_chronotype",)
    assert repair_plan.block_ids == ("dynamics:chronotype",)
    assert repair_plan.prompt_blocks[0].parameter_names == ("t0_sd_chronotype",)
    assert repair_plan.uses_repair_campaign is True
    assert (
        repair_plan.scope.reason
        == "Scale mismatch for monthly_eveningness_activity_timing Suggested fix: "
        "Adjust diffusion/drift priors to match data scale"
    )


def test_scale_mismatch_for_sparse_model_spec_routes_to_dynamics_block():
    causal_spec = _with_positive_indicator_polarity(
        {
            "latent": {
                "constructs": [
                    {
                        "name": "chronotype",
                        "role": "exogenous",
                        "temporal_status": "time_invariant",
                    },
                    {
                        "name": "sleep_quality",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "is_outcome": True,
                    },
                ],
                "edges": [
                    {"cause": "chronotype", "effect": "sleep_quality", "lagged": False},
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "monthly_eveningness_activity_timing",
                        "construct_name": "chronotype",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                        "observation_window": "1mo",
                    },
                    {
                        "name": "sleep_quality_search_count",
                        "construct_name": "sleep_quality",
                        "measurement_dtype": "count",
                        "aggregation": "sum",
                    },
                ],
            },
            "estimation": {
                "state_order": ["chronotype", "sleep_quality"],
                "edges": [
                    {"cause": "chronotype", "effect": "sleep_quality", "lagged": False},
                ],
                "induced_dependencies": [],
            },
        }
    )
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    runtime = make_stage4_runtime(plan)
    active_block = _require_plan_block(plan, "dynamics:sleep_quality")

    repair_plan = classify_prior_failure_blocks(
        plan,
        active_block,
        AssemblyValidation(
            normalized_model_spec={
                "likelihoods": [
                    {
                        "variable": "monthly_eveningness_activity_timing",
                        "distribution": "gaussian",
                        "link": "identity",
                        "centered": True,
                    }
                ],
                "parameters": [
                    {
                        "name": "t0_mean_chronotype",
                        "role": "initial_state_mean",
                    },
                    {
                        "name": "t0_sd_chronotype",
                        "role": "initial_state_sd",
                    },
                    {
                        "name": "beta_chronotype_sleep_quality",
                        "role": "fixed_effect",
                    },
                ],
            },
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="scale_monthly_eveningness_activity_timing",
                    is_valid=False,
                    code="scale_mismatch",
                    origin="prior_predictive",
                    issue="Scale mismatch for monthly_eveningness_activity_timing",
                    suggested_adjustment="Adjust diffusion/drift priors to match data scale",
                )
            ],
        ),
        runtime,
    )

    assert repair_plan.scope.scope_kind == "direct_writer_blocks"
    assert repair_plan.scope.parameter_names == ("t0_sd_chronotype",)
    assert repair_plan.block_ids == ("dynamics:chronotype",)
    assert repair_plan.prompt_blocks[0].parameter_names == ("t0_sd_chronotype",)
    assert repair_plan.uses_repair_campaign is True


def test_scale_mismatch_escalates_to_global_prior_review_after_local_scope_exhausts():
    causal_spec = _with_positive_indicator_polarity(
        {
            "latent": {
                "constructs": [
                    {
                        "name": "chronotype",
                        "role": "exogenous",
                        "temporal_status": "time_invariant",
                    },
                    {
                        "name": "sleep_quality",
                        "role": "endogenous",
                        "temporal_status": "time_varying",
                        "is_outcome": True,
                    },
                ],
                "edges": [
                    {"cause": "chronotype", "effect": "sleep_quality", "lagged": False},
                ],
            },
            "measurement": {
                "model_clock": "1d",
                "indicators": [
                    {
                        "name": "monthly_eveningness_activity_timing",
                        "construct_name": "chronotype",
                        "measurement_dtype": "continuous",
                        "aggregation": "mean",
                        "observation_window": "1mo",
                    },
                    {
                        "name": "sleep_quality_search_count",
                        "construct_name": "sleep_quality",
                        "measurement_dtype": "count",
                        "aggregation": "sum",
                    },
                ],
            },
            "estimation": {
                "state_order": ["chronotype", "sleep_quality"],
                "edges": [
                    {"cause": "chronotype", "effect": "sleep_quality", "lagged": False},
                ],
                "induced_dependencies": [],
            },
        }
    )
    validation = AssemblyValidation(
        normalized_model_spec={
            "likelihoods": [
                {
                    "variable": "monthly_eveningness_activity_timing",
                    "distribution": "gaussian",
                    "link": "identity",
                    "centered": True,
                }
            ],
            "parameters": [
                {
                    "name": "t0_mean_chronotype",
                    "role": "initial_state_mean",
                    "construct": "chronotype",
                },
                {
                    "name": "t0_sd_chronotype",
                    "role": "initial_state_sd",
                    "construct": "chronotype",
                },
            ],
        },
        compile_ok=True,
        pp_checked=True,
        pp_valid=False,
        diagnostics=[
            PriorValidationResult(
                parameter="scale_monthly_eveningness_activity_timing",
                is_valid=False,
                code="scale_mismatch",
                origin="prior_predictive",
                issue="Scale mismatch for monthly_eveningness_activity_timing",
                suggested_adjustment="Adjust diffusion/drift priors to match data scale",
            )
        ],
    )
    skeleton = derive_deterministic_spec(causal_spec)
    plan = build_stage4_plan(causal_spec, skeleton)
    runtime = make_stage4_runtime(plan)
    active_block = _require_plan_block(plan, "dynamics:sleep_quality")

    local_repair_plan = classify_prior_failure_blocks(
        plan,
        active_block,
        validation,
        runtime,
    )
    runtime.domain.repair_campaign = Stage4RepairCampaignState(
        failure_family_key=local_repair_plan.scope.failure_family,
        scope_kind=local_repair_plan.scope.scope_kind,
        scope_key=local_repair_plan.scope.scope_key,
        scope_rank=local_repair_plan.scope.scope_rank,
        scope_block_ids=local_repair_plan.block_ids,
        prompt_blocks_by_id={block.id: block for block in local_repair_plan.prompt_blocks},
        attempts_at_scope=2,
    )

    escalated_repair_plan = classify_prior_failure_blocks(
        plan,
        active_block,
        validation,
        runtime,
    )

    assert escalated_repair_plan.scope.scope_kind == "global_prior_review"
    assert escalated_repair_plan.block_ids == ("review:prior_system",)


def test_prior_failure_classification_raises_without_concrete_reason():
    causal_spec, skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
    del causal_spec, skeleton
    active_block = _require_plan_block(plan, "dynamics:sleep")

    with pytest.raises(
        ValueError,
        match="requires a concrete reason",
    ):
        classify_prior_failure_blocks(
            plan,
            active_block,
            AssemblyValidation(
                normalized_model_spec={"likelihoods": [], "parameters": []},
                compile_ok=True,
                pp_checked=True,
                pp_valid=False,
                diagnostics=[
                    PriorValidationResult(
                        parameter="rho_sleep",
                        is_valid=False,
                        code="unspecified",
                        origin="prior_predictive",
                        issue=None,
                        suggested_adjustment=None,
                        related_parameters=["rho_sleep"],
                    )
                ],
            ),
            runtime,
        )

