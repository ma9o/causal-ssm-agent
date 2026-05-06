"""Stage 4 reducer, session, and scripted flow tests."""

from tests.stages.stage4._support import (
    _ORDINAL_LEVELS,
    Any,
    AssemblyValidation,
    GenerateConfig,
    PriorPathologyCertificate,
    PriorRepairScope,
    PriorValidationResult,
    ResolvedRepairScope,
    SimpleNamespace,
    Stage4AcceptedArtifacts,
    Stage4DomainState,
    Stage4FatalSubmissionError,
    Stage4FrontierBlock,
    Stage4RepairCampaignState,
    Stage4Runtime,
    Stage4Skeleton,
    Tool,
    _accept_default_model_configuration,
    _activity_loading_prior_bundle,
    _activity_measurement_prior_bundle,
    _apply_stage4_step_and_capture,
    _await_string,
    _current_stage4_state,
    _load_resumable_stage4_runtime,
    _make_plan,
    _make_runtime,
    _make_scripted_stage4_generate,
    _make_scripted_stage4_generate_by_block,
    _make_stage4_deps,
    _make_stage4_global_repair_spec,
    _make_stage4_mechanics_context,
    _make_stage4_mechanics_spec,
    _make_stage4_no_model_block_spec,
    _make_stage4_session,
    _make_stage4_session_factory,
    _make_stage4_two_effect_spec,
    _make_stub_grounding_result,
    _require_active_plan_block,
    _require_plan_block,
    _require_trace,
    _set_done_cursor,
    _set_runtime_block,
    _stage4_submit_tool_args,
    _stage4_submit_tool_name,
    _stage4_test_payload,
    _stub_stage4_repair_barrier_success,
    _validate_stage4_runtime_checkpoint,
    _with_positive_indicator_polarity,
    asyncio,
    build_model_spec_from_decisions,
    build_prior_cards,
    build_stage4_plan,
    classify_prior_failure_blocks,
    compute_stage4_validate_step,
    compute_stage4_validate_step_with_transitions,
    deepcopy,
    derive_deterministic_spec,
    execute_tools,
    format_stage4_plan_status,
    get_active_plan_block,
    get_stage4_block_handler,
    get_stage4_phase,
    json,
    make_generate_fn,
    make_stage4_runtime,
    np,
    pl,
    pytest,
    run_stage4,
)


def test_run_stage4_returns_captured_validation(monkeypatch):
    """The last successful validation should be carried into materialization."""
    skeleton = SimpleNamespace(
        all_params=[],
        loading_params=[],
        resolved_likelihoods=[],
        ambiguous_indicators=[],
    )
    validation = AssemblyValidation(pp_checked=True, pp_valid=True)
    capture = {
        "model_spec": {"likelihoods": [{"variable": "mood_score"}]},
        "authored_priors": {"rho_mood": {"distribution": "Beta"}},
        "validation": validation,
    }

    def stub_derive_deterministic_spec(causal_spec):
        del causal_spec
        return skeleton

    def stub_build_model_topology(causal_spec):
        del causal_spec
        return {}

    def stub_build_distribution_cards(causal_spec, indicator_audits, skeleton):
        del causal_spec, indicator_audits, skeleton
        return []

    def stub_build_construct_scale_cards(causal_spec, indicator_audits, skeleton):
        del causal_spec, indicator_audits, skeleton
        return []

    def stub_build_prior_cards(causal_spec, skeleton):
        del causal_spec, skeleton
        return []

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.derive_deterministic_spec",
        stub_derive_deterministic_spec,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_model_topology",
        stub_build_model_topology,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_distribution_cards",
        stub_build_distribution_cards,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_construct_scale_cards",
        stub_build_construct_scale_cards,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_prior_cards",
        stub_build_prior_cards,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.build_stage4_plan",
        lambda _causal_spec, _skeleton: _make_plan(),
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.agentic.stage4_agent_loop.make_stage4_runtime",
        lambda _plan: Stage4Runtime(
            domain=Stage4DomainState(
                done=True,
                accepted=Stage4AcceptedArtifacts(
                    model_spec=capture["model_spec"],
                    authored_priors=capture["authored_priors"],
                    validation=validation,
                ),
            )
        ),
    )

    def stub_stage4_grounding(*_args, **_kwargs):
        return _make_stub_grounding_result(capture, "VALID")

    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
        stub_stage4_grounding,
    )

    async def fake_generate(messages, tools, label=None):
        del messages, tools, label
        pytest.fail("generate should not run when Stage 4 auto-completes before prompting")

    result = asyncio.run(
        run_stage4(
            causal_spec={},
            question="How can I be more productive?",
            data_for_model=pl.DataFrame(),
            indicator_audits={},
            session_factory=_make_stage4_session_factory(fake_generate),
            enable_literature=False,
        )
    )

    assert result.validation is validation


def test_materialize_override_stage4_marks_missing_compiled_ssm_as_failure(monkeypatch):
    from causal_ssm_agent.flows.stage_runtime import PipelineContext
    from causal_ssm_agent.flows.stages.stage4.definition import build_stage4_definition

    monkeypatch.setattr(
        "causal_ssm_agent.flows.run_store.find_run_artifact",
        lambda *_args, **_kwargs: "ignored.parquet",
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.run_store.load_parquet",
        lambda *_args, **_kwargs: pl.DataFrame(),
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.run_store.save_json",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "causal_ssm_agent.flows.stages.stage4.assembly.materialize_stage4_result",
        lambda **_kwargs: {
            "model_spec": {"likelihoods": [], "parameters": []},
            "authored_priors": {},
            "resolved_priors": [],
            "validation_warnings": [],
        },
    )

    ctx = PipelineContext(
        workspace_id="workspace",
        prefect_run_id="run",
        question="question",
        lit_enabled=False,
        inference_method=None,
        supported_overrides={},
        openrouter_api_key=None,
        openrouter_access_mode=None,
    )
    states = {
        "stage-1b": SimpleNamespace(causal_spec=SimpleNamespace(model_dump=lambda: {})),
        "stage-3": SimpleNamespace(indicators={}),
    }

    adapter = build_stage4_definition().override_adapter
    assert adapter is not None
    contract = adapter.materialize(
        {"model_spec": {"likelihoods": [], "parameters": []}, "authored_priors": {}},
        ctx,
        states,
    )

    assert contract.outcome == "fail"
    assert contract.fail_reason == "model_compile_failed"


class TestStage4Mechanics:
    def test_format_plan_status_exposes_effect_row_budget(self):
        causal_spec, skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )
        runtime.domain.draft_model.distribution_choices["steps"] = {
            "variable": "steps",
            "distribution": "poisson",
            "link": "log",
            "reasoning": "Step counts are nonnegative integers.",
        }
        model_spec, errors = build_model_spec_from_decisions(
            runtime.domain.draft_model,
            skeleton,
        )
        assert model_spec is not None
        assert errors == []
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec=model_spec,
            authored_priors={
                "obs_sd_activity_vas": {
                    "parameter": "obs_sd_activity_vas",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "activity VAS measurement noise",
                },
                "obs_sd_steps": {
                    "parameter": "obs_sd_steps",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "steps measurement noise",
                },
                "lambda_activity_vas_activity": {
                    "parameter": "lambda_activity_vas_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.4},
                    "sources": [],
                    "reasoning": "measurement prior",
                },
                "manifest_mean_steps": {
                    "parameter": "manifest_mean_steps",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 1.0},
                    "sources": [],
                    "reasoning": "steps observation intercept",
                },
                "obs_ordered_base": {
                    "parameter": "obs_ordered_base",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 1.0},
                    "sources": [],
                    "reasoning": "ordered-threshold location prior",
                },
                "sigma_activity": {
                    "parameter": "sigma_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "activity residual scale",
                },
                "rho_sleep": {
                    "parameter": "rho_sleep",
                    "distribution": "Beta",
                    "params": {"alpha": 4.0, "beta": 3.0},
                    "sources": [],
                    "reasoning": "sleep persistence",
                },
                "sigma_sleep": {
                    "parameter": "sigma_sleep",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "sleep residual scale",
                },
            },
        )
        _set_runtime_block(plan, runtime, "effects:sleep")
        block = _require_plan_block(plan, "effects:sleep")

        status = format_stage4_plan_status(
            plan,
            runtime,
            block,
            get_stage4_block_handler(block.kind),
            causal_spec=causal_spec,
        )

        assert "stability budget source" in status
        assert "target row budget" in status
        assert "remaining headroom" in status

    @pytest.mark.parametrize(
        ("payload", "expected_feedback"),
        [
            (
                {
                    "variable": "sleep",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "wrong indicator variable",
                },
                "proposal variable must be `steps`",
            ),
            (
                {
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "missing variable",
                },
                "VALIDATION ERRORS:",
            ),
        ],
    )
    def test_compute_stage4_validate_step_rejects_invalid_indicator_payloads(
        self,
        payload,
        expected_feedback,
    ):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )

        stage_output, feedback = compute_stage4_validate_step(
            payload,
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=data_for_model,
                stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                    "grounding should not run for invalid submissions"
                ),
            ),
        )

        assert stage_output is None
        assert expected_feedback in feedback
        assert runtime.interaction.last_validation_packet is not None
        assert runtime.interaction.last_validation_packet.model_feedback == feedback
        assert runtime.domain.draft_model.distribution_choices == {}
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"

    def test_compute_stage4_validate_step_reopens_model_block_when_model_lock_fails(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )

        indicator_payload = {
            "block_id": "indicator:steps",
            "block_kind": "indicator_decision",
            "proposal": {
                "variable": "steps",
                "distribution": "poisson",
                "link": "log",
                "reasoning": "Step counts are nonnegative integers.",
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            assert current == {}
            assert "model_spec" in data
            model_spec = data["model_spec"]
            return {
                "validation": AssemblyValidation(
                    normalized_model_spec=model_spec,
                    compile_ok=False,
                    compile_error="steps support mismatch",
                )
            }, "COMPILE ERROR:\nsteps support mismatch"

        stage_output, feedback = _apply_stage4_step_and_capture(
            indicator_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback == "COMPILE ERROR:\nsteps support mismatch"
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"
        assert runtime.domain.block_status["indicator:steps"] == "reopened"
        assert runtime.interaction.last_validation_packet is not None
        assert runtime.interaction.last_validation_packet.model_feedback == feedback
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"
        assert runtime.domain.accepted.as_current() == {}

    def test_compute_stage4_validate_step_emits_indicator_last_state_transitions(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )

        indicator_payload = {
            "block_id": "indicator:steps",
            "block_kind": "indicator_decision",
            "proposal": {
                "variable": "steps",
                "distribution": "poisson",
                "link": "log",
                "reasoning": "Step counts are nonnegative integers.",
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            assert current == {}
            assert "model_spec" in data
            model_spec = data["model_spec"]
            return {
                "validation": AssemblyValidation(
                    normalized_model_spec=model_spec,
                    compile_ok=False,
                    compile_error="steps support mismatch",
                )
            }, "COMPILE ERROR:\nsteps support mismatch"

        stage_output, feedback, transitions = compute_stage4_validate_step_with_transitions(
            _stage4_test_payload(indicator_payload),
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=data_for_model,
                stage4_grounding_fn=stub_stage4_grounding,
            ),
        )

        assert stage_output is not None
        assert feedback == "COMPILE ERROR:\nsteps support mismatch"
        assert transitions == (
            {
                "block_id": "indicator:steps",
                "status": "accepted",
                "detail_kind": "indicator_choice",
                "variable": "steps",
                "distribution": "poisson",
                "link": "log",
                "reasoning": "Step counts are nonnegative integers.",
            },
            {
                "block_id": "indicator:steps",
                "status": "reopened",
                "detail_kind": "revision",
                "reason": "steps support mismatch",
                "scope_kind": "compile_local",
            },
        )

    def test_compile_failure_route_uses_structured_manifest_diagnostics_without_exact_feedback_match(
        self,
    ):
        from causal_ssm_agent.flows.stages.stage4.agentic.stage4_repair import (
            classify_compile_failure_route,
        )

        causal_spec, skeleton, plan, _runtime, _data_for_model = _make_stage4_mechanics_context()
        del causal_spec, skeleton, _runtime, _data_for_model

        repair_plan = classify_compile_failure_route(
            plan,
            _require_plan_block(plan, "measurement:activity"),
            "Compile error: validator-owned manifest failure.",
            validation=AssemblyValidation(
                compile_ok=False,
                compile_error="Compile error: validator-owned manifest failure.",
                diagnostics=[
                    PriorValidationResult(
                        parameter="prior_predictive",
                        is_valid=False,
                        code="support_violation",
                        origin="compile",
                        issue="Manifest support is incompatible with the chosen likelihood.",
                        bad_manifest_names=["steps"],
                    )
                ],
            ),
        )

        assert repair_plan.scope.scope_kind == "compile_local"
        assert repair_plan.block_ids == ("indicator:steps",)
        assert repair_plan.uses_repair_campaign is False

    def test_model_lock_failure_block_ids_cover_configuration_and_indicator_attribution(self):
        from causal_ssm_agent.flows.stages.stage4.agentic.stage4_reducer import (
            _model_lock_failure_block_ids,
        )

        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()
        del causal_spec, skeleton, data_for_model

        block_ids = _model_lock_failure_block_ids(
            plan,
            runtime.domain.draft_model,
            [
                "'equilibrium_forcing' must be a boolean",
                "missing distribution_choice for ambiguous indicator 'steps'",
            ],
            None,
        )

        assert block_ids == ("model:configuration", "indicator:steps")

    def test_compute_stage4_validate_step_keeps_effect_block_when_only_budget_is_tight(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )
        runtime.domain.draft_model.distribution_choices["steps"] = {
            "variable": "steps",
            "distribution": "poisson",
            "link": "log",
            "reasoning": "Step counts are nonnegative integers.",
        }
        model_spec, errors = build_model_spec_from_decisions(
            runtime.domain.draft_model,
            skeleton,
        )
        assert model_spec is not None
        assert errors == []
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec=model_spec,
            authored_priors={
                "obs_sd_activity_vas": {
                    "parameter": "obs_sd_activity_vas",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "activity VAS measurement noise",
                },
                "obs_sd_steps": {
                    "parameter": "obs_sd_steps",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "steps measurement noise",
                },
                "lambda_activity_vas_activity": {
                    "parameter": "lambda_activity_vas_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.4},
                    "sources": [],
                    "reasoning": "measurement prior",
                },
                "manifest_mean_steps": {
                    "parameter": "manifest_mean_steps",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 1.0},
                    "sources": [],
                    "reasoning": "steps observation intercept",
                },
                "obs_ordered_base": {
                    "parameter": "obs_ordered_base",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 1.0},
                    "sources": [],
                    "reasoning": "ordered-threshold location prior",
                },
                "sigma_activity": {
                    "parameter": "sigma_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "activity residual scale",
                },
                "rho_sleep": {
                    "parameter": "rho_sleep",
                    "distribution": "Beta",
                    "params": {"alpha": 20.0, "beta": 1.0},
                    "sources": [],
                    "reasoning": "very persistent sleep state",
                },
                "sigma_sleep": {
                    "parameter": "sigma_sleep",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "sleep residual scale",
                },
            },
        )
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.25, "sigma": 0.05},
                        "sources": [],
                        "reasoning": "strong incoming sleep effect",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=False,
                    pp_valid=True,
                ),
            }, "BLOCK ACCEPTED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback == "BLOCK ACCEPTED"
        assert runtime.domain.repair_campaign is None
        assert runtime.domain.block_status["effects:sleep"] == "accepted"
        assert "beta_activity_sleep" in runtime.domain.accepted.authored_priors

    def test_compute_stage4_validate_step_reopens_dynamics_block_on_partial_drift_guard(
        self, monkeypatch
    ):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )
        runtime.domain.draft_model.distribution_choices["steps"] = {
            "variable": "steps",
            "distribution": "poisson",
            "link": "log",
            "reasoning": "Step counts are nonnegative integers.",
        }
        model_spec, errors = build_model_spec_from_decisions(
            runtime.domain.draft_model,
            skeleton,
        )
        assert model_spec is not None
        assert errors == []
        _set_runtime_block(plan, runtime, "dynamics:sleep")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec=model_spec,
            authored_priors={
                "lambda_activity_vas_activity": {
                    "parameter": "lambda_activity_vas_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.4},
                    "sources": [],
                    "reasoning": "measurement prior",
                },
                "sigma_activity": {
                    "parameter": "sigma_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "sources": [],
                    "reasoning": "activity residual scale",
                },
            },
        )
        dynamics_payload = {
            "block_id": "dynamics:sleep",
            "block_kind": "dynamics_prior",
            "proposal": {
                "priors": {
                    "rho_sleep": {
                        "parameter": "rho_sleep",
                        "distribution": "Beta",
                        "params": {"alpha": 20.0, "beta": 1.0},
                        "sources": [],
                        "reasoning": "very persistent sleep state",
                    },
                    "sigma_sleep": {
                        "parameter": "sigma_sleep",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.5},
                        "sources": [],
                        "reasoning": "sleep residual scale",
                    },
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=False,
                    pp_valid=True,
                ),
            }, "BLOCK ACCEPTED"

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_partial_drift.validate_dynamics_block_partial_drift",
            lambda **_kwargs: (
                PriorValidationResult(
                    parameter="rho_sleep",
                    is_valid=False,
                    code="partial_dynamics_budget_exhausted",
                    origin="prior_predictive",
                    issue="The sleep row has no conservative damping headroom yet.",
                    suggested_adjustment="Tighten rho_sleep toward faster decay.",
                    related_parameters=["sigma_sleep"],
                    failure_stage="latent_dynamics",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dynamics_stability",
                        primary_score=0.1,
                    ),
                ),
                "\n".join(
                    [
                        "PARTIAL DRIFT CHECK FAILED:",
                        "- target row `sleep` has no conservative damping headroom yet",
                        "- revise this dynamics block before eliciting downstream effect priors",
                    ]
                ),
            ),
        )

        stage_output, feedback = _apply_stage4_step_and_capture(
            dynamics_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is None
        assert feedback.startswith("PARTIAL DRIFT CHECK FAILED:")
        assert runtime.domain.repair_campaign is not None
        assert runtime.domain.repair_campaign.scope_block_ids == ("dynamics:sleep",)
        assert runtime.domain.block_status["dynamics:sleep"] == "reopened"
        assert _require_active_plan_block(plan, runtime).id == "dynamics:sleep"
        assert "rho_sleep" not in runtime.domain.accepted.authored_priors
        assert "sigma_sleep" not in runtime.domain.accepted.authored_priors
        assert runtime.interaction.last_validation_packet is not None
        assert runtime.interaction.last_validation_packet.status == "partial_drift_failure"
        assert runtime.interaction.last_validation_packet.failing_parameters == ("rho_sleep",)
        assert runtime.interaction.last_validation_packet.coupled_parameters == ("sigma_sleep",)
        assert runtime.interaction.last_validation_packet.diagnostic_codes == (
            "partial_dynamics_budget_exhausted",
        )

    def test_global_review_can_reopen_model_block_set(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )
        _set_runtime_block(plan, runtime, "review:model_spec")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "locked"}]}
        )
        runtime.domain.block_status["indicator:steps"] = "accepted"
        runtime.domain.block_status["review:model_spec"] = "pending"

        stage_output, feedback = compute_stage4_validate_step(
            {
                "decision": "reopen",
                "reopen_block_ids": ["indicator:steps"],
                "reasoning": "The count likelihood should be reconsidered.",
            },
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=data_for_model,
                stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                    "grounding should not run for review-only reopen decisions"
                ),
            ),
        )

        assert stage_output is None
        assert "MODEL REVIEW REOPENED" in feedback
        assert "`indicator:steps`" in feedback
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"
        assert get_stage4_phase(runtime, plan=plan) == "model_decisions"
        assert runtime.domain.block_status["indicator:steps"] == "reopened"

    def test_global_review_allows_reopening_more_than_three_model_blocks(self):
        model_blocks = (
            Stage4FrontierBlock(
                id="indicator:a",
                kind="indicator_decision",
                label="Indicator a",
                variable_names=("a",),
            ),
            Stage4FrontierBlock(
                id="indicator:b",
                kind="indicator_decision",
                label="Indicator b",
                variable_names=("b",),
            ),
            Stage4FrontierBlock(
                id="indicator:c",
                kind="indicator_decision",
                label="Indicator c",
                variable_names=("c",),
            ),
            Stage4FrontierBlock(
                id="indicator:d",
                kind="indicator_decision",
                label="Indicator d",
                variable_names=("d",),
            ),
        )
        review_block = Stage4FrontierBlock(
            id="review:model_spec",
            kind="global_review",
            label="Review",
            payload={"reopenable_block_ids": tuple(block.id for block in model_blocks)},
        )
        plan = _make_plan(model_blocks=model_blocks, review_block=review_block)
        runtime = _make_runtime(
            plan,
            phase="global_review",
            active_block_id="review:model_spec",
            accepted=Stage4AcceptedArtifacts(model_spec={"parameters": [{"name": "locked"}]}),
        )
        for block in model_blocks:
            runtime.domain.block_status[block.id] = "accepted"
        runtime.domain.block_status["review:model_spec"] = "pending"

        stage_output, feedback = compute_stage4_validate_step(
            {
                "decision": "reopen",
                "reopen_block_ids": [block.id for block in model_blocks],
                "reasoning": "These measurement decisions need to be reconsidered together.",
            },
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec={},
                skeleton=Stage4Skeleton(),
                stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                    "grounding should not run for review-only reopen decisions"
                ),
            ),
        )

        assert stage_output is None
        assert "MODEL REVIEW REOPENED" in feedback
        assert "`indicator:a`, `indicator:b`, `indicator:c`, `indicator:d`" in feedback
        assert _require_active_plan_block(plan, runtime).id == "indicator:a"
        assert get_stage4_phase(runtime, plan=plan) == "model_decisions"
        for block in model_blocks:
            assert runtime.domain.block_status[block.id] == "reopened"

    def test_compute_stage4_validate_step_reopens_indicator_on_support_mismatch(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {"variable": "steps", "distribution": "poisson", "link": "log"},
                ],
                "parameters": [
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "lambda_activity_vas_activity"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
            },
        )
        _set_runtime_block(plan, runtime, "effects:sleep")
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "effect prior with support issue",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="model_build",
                            is_valid=False,
                            code="model_build",
                            origin="prior_predictive",
                            issue=(
                                "Observation support check failed:\n"
                                "- 'steps' uses gamma emission but observations are outside support"
                            ),
                            suggested_adjustment="Fix the emission support",
                        )
                    ],
                ),
            }, "PRIOR PREDICTIVE CHECKS FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback.startswith("PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED")
        assert "Observation support check failed" in feedback
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"
        assert runtime.domain.block_status["indicator:steps"] == "reopened"
        assert "beta_activity_sleep" in runtime.domain.accepted.authored_priors
        assert _require_active_plan_block(plan, runtime).id == "indicator:steps"

    def test_compute_stage4_validate_step_accepts_correlation_and_reopens_dynamics_scope(self):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _accept_default_model_configuration(
            causal_spec=causal_spec,
            skeleton=skeleton,
            plan=plan,
            runtime=runtime,
        )
        _set_runtime_block(plan, runtime, "correlation:tau_U")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "tau_U"},
                ],
                "initialization_policy": "stationary",
                "observation_intercept_policy": "free",
                "equilibrium_forcing": False,
            },
            authored_priors={
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
        )
        correlation_payload = {
            "block_id": "correlation:tau_U",
            "block_kind": "correlation_prior",
            "proposal": {
                "priors": {
                    "tau_U": {
                        "parameter": "tau_U",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.2},
                        "sources": [],
                        "reasoning": "baseline-factor prior",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                        ),
                        PriorValidationResult(
                            parameter="dynamics_stability",
                            is_valid=False,
                            code="dynamics_stability",
                            origin="prior_predictive",
                            issue="Unstable dynamics: 32/50 prior draws have unstable drift",
                            suggested_adjustment=(
                                "Increase base damping by tightening rho priors toward lower "
                                "baseline persistence"
                            ),
                            repair_scope=PriorRepairScope(
                                kind="dynamics_scc",
                                construct_names=["sleep"],
                            ),
                        ),
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            correlation_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback.startswith("PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED")
        assert "Unstable dynamics" in feedback
        assert runtime.domain.block_status["correlation:tau_U"] == "accepted"
        assert runtime.domain.block_status["dynamics:sleep"] == "reopened"
        assert runtime.domain.block_status["effects:sleep"] == "pending"
        assert runtime.domain.repair_campaign is not None
        assert runtime.domain.repair_campaign.scope_key == "validator_scope:sleep"
        assert runtime.domain.repair_campaign.scope_block_ids == ("dynamics:sleep",)
        assert _require_active_plan_block(plan, runtime).id == "dynamics:sleep"
        assert "tau_U" in runtime.domain.accepted.authored_priors
        assert _require_active_plan_block(plan, runtime).id == "dynamics:sleep"

    def test_compute_stage4_validate_step_emits_prior_and_revision_last_state_transitions(self):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _accept_default_model_configuration(
            causal_spec=causal_spec,
            skeleton=skeleton,
            plan=plan,
            runtime=runtime,
        )
        _set_runtime_block(plan, runtime, "correlation:tau_U")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "tau_U"},
                ],
                "initialization_policy": "stationary",
                "observation_intercept_policy": "free",
                "equilibrium_forcing": False,
            },
            authored_priors={
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
        )
        correlation_payload = {
            "block_id": "correlation:tau_U",
            "block_kind": "correlation_prior",
            "proposal": {
                "priors": {
                    "tau_U": {
                        "parameter": "tau_U",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.2},
                        "sources": [],
                        "reasoning": "baseline-factor prior",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                        ),
                        PriorValidationResult(
                            parameter="dynamics_stability",
                            is_valid=False,
                            code="dynamics_stability",
                            origin="prior_predictive",
                            issue="Unstable dynamics: 32/50 prior draws have unstable drift",
                            suggested_adjustment=(
                                "Increase base damping by tightening rho priors toward lower "
                                "baseline persistence"
                            ),
                            repair_scope=PriorRepairScope(
                                kind="dynamics_scc",
                                construct_names=["sleep"],
                            ),
                        ),
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback, transitions = compute_stage4_validate_step_with_transitions(
            _stage4_test_payload(correlation_payload),
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                stage4_grounding_fn=stub_stage4_grounding,
            ),
        )

        assert stage_output is not None
        assert feedback.startswith("PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED")
        assert "Unstable dynamics" in feedback
        assert transitions[0] == {
            "block_id": "correlation:tau_U",
            "status": "accepted",
            "detail_kind": "prior_bundle",
            "parameter_names": ["tau_U"],
            "priors": [
                {
                    "parameter": "tau_U",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.2},
                    "reasoning": "baseline-factor prior",
                }
            ],
        }
        assert transitions[1]["block_id"] == "dynamics:sleep"
        assert transitions[1]["status"] == "reopened"
        assert transitions[1]["detail_kind"] == "revision"
        assert transitions[1]["scope_kind"] == "validator_scope"
        assert (
            transitions[1]["reason"]
            == "Unstable dynamics: 32/50 prior draws have unstable drift Suggested "
            "fix: Increase base damping by tightening rho priors toward lower baseline persistence"
        )

    def test_compute_stage4_validate_step_emits_accepted_transition_for_barrier_campaign_repair(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _accept_default_model_configuration(
            causal_spec=causal_spec,
            skeleton=skeleton,
            plan=plan,
            runtime=runtime,
        )
        repair_block = _require_plan_block(plan, "correlation:tau_U")
        next_block = _require_plan_block(plan, "effects:sleep")
        _set_runtime_block(plan, runtime, repair_block.id)
        runtime.domain.block_status[repair_block.id] = "reopened"
        runtime.domain.block_status[next_block.id] = "reopened"
        runtime.domain.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(("prior_predictive_nonfinite_samples",), (), ("activity", "sleep")),
            scope_kind="validator_scope",
            scope_key="validator_scope:sleep",
            scope_rank=0,
            scope_block_ids=(repair_block.id, next_block.id),
            prompt_blocks_by_id={
                repair_block.id: repair_block,
                next_block.id: next_block,
            },
            completed_block_ids=frozenset(),
            attempts_at_scope=1,
            best_certificate=None,
        )
        repair_payload = {
            "block_id": repair_block.id,
            "block_kind": "correlation_prior",
            "proposal": {
                "priors": {
                    "tau_U": {
                        "parameter": "tau_U",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.15},
                        "sources": [],
                        "reasoning": "repair tightened the baseline-factor prior",
                    }
                }
            },
        }

        seen_skip_ppc: list[bool] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            seen_skip_ppc.append(_kwargs["skip_ppc"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=False,
                ),
            }, "VALID"

        stage_output, feedback, transitions = compute_stage4_validate_step_with_transitions(
            _stage4_test_payload(repair_payload),
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                stage4_grounding_fn=stub_stage4_grounding,
            ),
        )

        assert stage_output is not None
        assert feedback.startswith("REPAIR CAMPAIGN PROGRESS:\n")
        assert seen_skip_ppc == [True]
        assert transitions == (
            {
                "block_id": "correlation:tau_U",
                "status": "accepted",
                "detail_kind": "prior_bundle",
                "parameter_names": ["tau_U"],
                "priors": [
                    {
                        "parameter": "tau_U",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.15},
                        "reasoning": "repair tightened the baseline-factor prior",
                    }
                ],
            },
        )
        assert runtime.domain.block_status[repair_block.id] == "accepted"
        assert runtime.domain.repair_campaign is not None
        assert runtime.domain.repair_campaign.completed_block_ids == frozenset((repair_block.id,))
        assert _require_active_plan_block(plan, runtime).id == next_block.id

    def test_compute_stage4_validate_step_reruns_ppc_after_final_barrier_repair(self, monkeypatch):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _accept_default_model_configuration(
            causal_spec=causal_spec,
            skeleton=skeleton,
            plan=plan,
            runtime=runtime,
        )
        repair_block = _require_plan_block(plan, "correlation:tau_U")
        final_block = _require_plan_block(plan, "effects:sleep")

        model_spec = {
            "likelihoods": [
                {
                    "variable": "activity_vas",
                    "distribution": "ordered_logistic",
                    "link": "logit",
                },
                {
                    "variable": "sleep_quality",
                    "distribution": "ordered_logistic",
                    "link": "logit",
                },
            ],
            "parameters": [
                {"name": "sigma_activity"},
                {"name": "rho_sleep"},
                {"name": "sigma_sleep"},
                {"name": "beta_activity_sleep"},
                {"name": "tau_U"},
            ],
            "initialization_policy": "stationary",
            "observation_intercept_policy": "free",
            "equilibrium_forcing": False,
        }
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec=model_spec,
            authored_priors={
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "tau_U": {"distribution": "HalfNormal"},
            },
        )
        _set_runtime_block(plan, runtime, final_block.id)
        runtime.domain.block_status[repair_block.id] = "accepted"
        runtime.domain.block_status[final_block.id] = "reopened"
        runtime.domain.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(("prior_predictive_nonfinite_samples",), (), ("activity", "sleep")),
            scope_kind="validator_scope",
            scope_key="validator_scope:sleep",
            scope_rank=0,
            scope_block_ids=(repair_block.id, final_block.id),
            prompt_blocks_by_id={
                repair_block.id: repair_block,
                final_block.id: final_block,
            },
            completed_block_ids=frozenset((repair_block.id,)),
            attempts_at_scope=1,
            best_certificate=None,
        )
        repair_payload = {
            "block_id": final_block.id,
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.15},
                        "sources": [],
                        "reasoning": "repair tightened the effect prior",
                    }
                }
            },
        }

        seen_skip_ppc: list[bool] = []
        barrier_validations: list[dict[str, Any]] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            seen_skip_ppc.append(kwargs["skip_ppc"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=False,
                    pp_valid=True,
                ),
            }, "VALID"

        def stub_validate_assembly(
            model_spec_arg,
            authored_priors_arg,
            _data_for_model,
            _indicator_audits,
            _causal_spec,
            *,
            skip_ppc=False,
        ):
            barrier_validations.append(
                {
                    "model_spec": model_spec_arg,
                    "authored_priors": dict(authored_priors_arg or {}),
                    "skip_ppc": skip_ppc,
                }
            )
            return AssemblyValidation(
                normalized_model_spec=model_spec_arg,
                compile_ok=True,
                pp_checked=True,
                pp_valid=False,
                diagnostics=[
                    PriorValidationResult(
                        parameter="beta_activity_sleep",
                        is_valid=False,
                        code="effect_still_too_wide",
                        origin="prior_predictive",
                        issue="Effect prior still too wide",
                        suggested_adjustment="Tighten beta_activity_sleep",
                    )
                ],
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.assembly.validate_assembly",
            stub_validate_assembly,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_partial_drift.validate_effect_block_partial_drift",
            lambda **_kwargs: None,
        )

        stage_output, feedback, _transitions = compute_stage4_validate_step_with_transitions(
            _stage4_test_payload(repair_payload),
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                stage4_grounding_fn=stub_stage4_grounding,
            ),
        )

        assert stage_output is not None
        assert feedback.startswith("PRIOR PREDICTIVE FEEDBACK:\n")
        assert seen_skip_ppc == [True]
        assert len(barrier_validations) == 1
        assert barrier_validations[0]["skip_ppc"] is False
        assert "beta_activity_sleep" in barrier_validations[0]["authored_priors"]
        assert runtime.domain.accepted.validation is not None
        assert runtime.domain.accepted.validation.pp_checked is True
        assert runtime.domain.accepted.validation.pp_valid is False
        assert runtime.interaction.last_validation_packet is not None
        assert runtime.interaction.last_validation_packet.status == "prior_predictive_failure"

    def test_compute_stage4_validate_step_synthesizes_bounded_repair_bundle_from_supporting_diagnostics(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
            },
        )
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "global repair trigger",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="drift_offdiag",
                            is_valid=True,
                            code="dt_ct_approximation_warning",
                            origin="compile",
                            severity="warning",
                            issue="off-diagonal drift is large relative to damping",
                            related_parameters=["beta_activity_sleep"],
                        ),
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                            related_parameters=["drift_offdiag"],
                            supporting_codes=["dt_ct_approximation_warning"],
                        ),
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback.startswith("REPAIR CAMPAIGN ACTIVE:\n")
        assert "scope: `local_drift_motif:activity|sleep|beta_activity_sleep`" in feedback
        assert runtime.domain.block_status["effects:sleep"] == "accepted"
        assert runtime.domain.block_status["dynamics:activity"] == "reopened"
        assert runtime.domain.block_status["dynamics:sleep"] == "reopened"
        assert _require_active_plan_block(plan, runtime).id == "dynamics:activity"
        assert get_stage4_phase(runtime, plan=plan) == "prior_blocks"
        assert "beta_activity_sleep" in runtime.domain.accepted.authored_priors

    def test_prior_failure_classification_allows_same_scope_retry_before_default_cap(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.domain.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(
                ("prior_predictive_nonfinite_samples",),
                ("dt_ct_approximation_warning",),
                ("activity", "sleep"),
            ),
            scope_kind="local_drift_motif",
            scope_key="local_drift_motif:activity|sleep|beta_activity_sleep",
            scope_rank=0,
            scope_block_ids=("dynamics:activity", "dynamics:sleep", "effects:sleep"),
            completed_block_ids=frozenset(("dynamics:activity", "dynamics:sleep", "effects:sleep")),
            attempts_at_scope=1,
            best_certificate=PriorPathologyCertificate(
                kind="nonfinite_samples",
                primary_score=1.0,
            ),
        )

        validation = AssemblyValidation(
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="drift_offdiag",
                    is_valid=True,
                    code="dt_ct_approximation_warning",
                    origin="compile",
                    severity="warning",
                    issue="off-diagonal drift is large relative to damping",
                    related_parameters=["beta_activity_sleep"],
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dt_ct_approximation",
                        primary_score=0.30,
                    ),
                ),
                PriorValidationResult(
                    parameter="prior_predictive",
                    is_valid=False,
                    code="prior_predictive_nonfinite_samples",
                    origin="prior_predictive",
                    issue="NaN/Inf detected in sample sites: observations",
                    suggested_adjustment="Check for degenerate priors",
                    related_parameters=["drift_offdiag"],
                    supporting_codes=["dt_ct_approximation_warning"],
                    pathology_certificate=PriorPathologyCertificate(
                        kind="nonfinite_samples",
                        primary_score=0.75,
                    ),
                ),
            ],
        )

        repair_plan = classify_prior_failure_blocks(
            plan,
            _require_plan_block(plan, "effects:sleep"),
            validation,
            runtime,
        )

        assert repair_plan.scope.scope_kind == "local_drift_motif"
        assert repair_plan.scope.scope_key == "local_drift_motif:activity|sleep|beta_activity_sleep"

    def test_prior_failure_classification_uses_review_block_retry_budget_for_global_prior_review(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "review:prior_system")
        runtime.domain.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(("prior_predictive_observation_mean_overflow",), (), ()),
            scope_kind="global_prior_review",
            scope_key="global_prior_review:prior_system",
            scope_rank=3,
            scope_block_ids=("review:prior_system",),
            completed_block_ids=frozenset(("review:prior_system",)),
            attempts_at_scope=4,
            best_certificate=PriorPathologyCertificate(
                kind="nonfinite_samples",
                primary_score=0.5,
                secondary_score=4.0,
            ),
        )

        validation = AssemblyValidation(
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="prior_predictive",
                    is_valid=False,
                    code="prior_predictive_observation_mean_overflow",
                    origin="prior_predictive",
                    issue=(
                        "Predictive log-link mean overflow before observation sampling: "
                        "linear predictor exceeded the finite exp range."
                    ),
                    suggested_adjustment="Tighten the log-link priors",
                    bad_manifest_names=["late_evening_google_activity_count"],
                    pathology_certificate=PriorPathologyCertificate(
                        kind="nonfinite_samples",
                        primary_score=0.5,
                        secondary_score=4.0,
                    ),
                )
            ],
        )

        repair_plan = classify_prior_failure_blocks(
            plan,
            _require_plan_block(plan, "review:prior_system"),
            validation,
            runtime,
        )

        assert repair_plan.scope.scope_kind == "global_prior_review"
        assert repair_plan.scope.scope_key == "global_prior_review:prior_system"

    def test_failure_evidence_surface_owns_supporting_compile_context(self):
        from causal_ssm_agent.flows.stages.stage4.agentic.stage4_repair import (
            _localize_prior_failure,
            build_stage4_failure_evidence,
        )

        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)

        validation = AssemblyValidation(
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="drift_offdiag",
                    is_valid=True,
                    code="dt_ct_approximation_warning",
                    origin="compile",
                    severity="warning",
                    issue="off-diagonal drift is large relative to damping",
                    related_parameters=["beta_activity_sleep"],
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dt_ct_approximation",
                        primary_score=0.30,
                    ),
                ),
                PriorValidationResult(
                    parameter="dynamics_stability",
                    is_valid=False,
                    code="dynamics_stability",
                    origin="prior_predictive",
                    issue="Unstable reciprocal drift subsystem",
                    related_parameters=["beta_activity_sleep"],
                    supporting_codes=["dt_ct_approximation_warning"],
                    repair_scope=PriorRepairScope(
                        kind="dynamics_scc",
                        construct_names=["activity", "sleep"],
                    ),
                    failure_stage="latent_dynamics",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dynamics_stability",
                        primary_score=0.6,
                    ),
                ),
            ],
        )

        evidence = build_stage4_failure_evidence(plan, validation)
        assert evidence.diagnostic_codes == ("dynamics_stability",)
        assert evidence.supporting_codes == ("dt_ct_approximation_warning",)
        assert tuple(
            diagnostic.parameter for diagnostic in evidence.supporting_compile_diagnostics
        ) == ("drift_offdiag",)

        localization = _localize_prior_failure(plan, validation)
        assert localization.direct_parameters == ("beta_activity_sleep",)
        assert localization.supporting_parameters == ("beta_activity_sleep",)
        assert localization.construct_names == ("activity", "sleep")
        assert localization.pathology_certificate == PriorPathologyCertificate(
            kind="dynamics_stability",
            primary_score=0.6,
        )
        assert localization.reasons.validator == "Unstable reciprocal drift subsystem"

    def test_prior_failure_classification_prefers_lowest_rank_scope_over_validator_scope(self):
        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {
                            "name": "activity",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "sleep",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                    ],
                    "edges": [
                        {"cause": "activity", "effect": "sleep"},
                        {"cause": "sleep", "effect": "activity"},
                    ],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "activity_vas",
                            "construct_name": "activity",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Activity rating",
                            "aggregation": "mean",
                        },
                        {
                            "name": "sleep_quality",
                            "construct_name": "sleep",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Sleep quality rating",
                            "aggregation": "mean",
                        },
                    ],
                },
                "estimation": {
                    "state_order": ["activity", "sleep"],
                    "edges": [
                        {"cause": "activity", "effect": "sleep"},
                        {"cause": "sleep", "effect": "activity"},
                    ],
                    "induced_dependencies": [],
                },
            }
        )
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")

        validation = AssemblyValidation(
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="drift_offdiag",
                    is_valid=True,
                    code="dt_ct_approximation_warning",
                    origin="compile",
                    severity="warning",
                    related_parameters=["beta_activity_sleep"],
                ),
                PriorValidationResult(
                    parameter="dynamics_stability",
                    is_valid=False,
                    code="dynamics_stability",
                    origin="prior_predictive",
                    issue="Unstable reciprocal drift subsystem",
                    related_parameters=["beta_activity_sleep"],
                    supporting_codes=["dt_ct_approximation_warning"],
                    repair_scope=PriorRepairScope(
                        kind="dynamics_scc",
                        construct_names=["activity", "sleep"],
                    ),
                    failure_stage="latent_dynamics",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dynamics_stability",
                        primary_score=0.6,
                    ),
                ),
            ],
        )

        repair_plan = classify_prior_failure_blocks(
            plan,
            _require_plan_block(plan, "effects:sleep"),
            validation,
            runtime,
        )

        assert repair_plan.scope.scope_kind == "local_drift_motif"
        assert repair_plan.block_ids == ("dynamics:activity+sleep", "effects:sleep")

    def test_scc_repair_plan_narrows_effect_prompt_to_internal_scc_parameters(self):
        from causal_ssm_agent.flows.stages.stage4.agentic.stage4_repair import build_repair_plan

        causal_spec = _with_positive_indicator_polarity(
            {
                "latent": {
                    "constructs": [
                        {"name": "stress", "role": "exogenous", "temporal_status": "time_varying"},
                        {
                            "name": "activity",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                        },
                        {
                            "name": "sleep",
                            "role": "endogenous",
                            "temporal_status": "time_varying",
                            "is_outcome": True,
                        },
                    ],
                    "edges": [
                        {"cause": "stress", "effect": "sleep"},
                        {"cause": "activity", "effect": "sleep"},
                        {"cause": "sleep", "effect": "activity"},
                    ],
                },
                "measurement": {
                    "model_clock": "1d",
                    "indicators": [
                        {
                            "name": "stress_vas",
                            "construct_name": "stress",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Stress rating",
                            "aggregation": "mean",
                        },
                        {
                            "name": "activity_vas",
                            "construct_name": "activity",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Activity rating",
                            "aggregation": "mean",
                        },
                        {
                            "name": "sleep_quality",
                            "construct_name": "sleep",
                            "measurement_dtype": "ordinal",
                            "ordinal_levels": list(_ORDINAL_LEVELS),
                            "how_to_measure": "Sleep quality rating",
                            "aggregation": "mean",
                        },
                    ],
                },
                "estimation": {
                    "state_order": ["stress", "activity", "sleep"],
                    "edges": [
                        {"cause": "stress", "effect": "sleep"},
                        {"cause": "activity", "effect": "sleep"},
                        {"cause": "sleep", "effect": "activity"},
                    ],
                    "induced_dependencies": [],
                },
            }
        )
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)

        repair_plan = build_repair_plan(
            plan,
            ResolvedRepairScope(
                scope_kind="scc_drift_subsystem",
                scope_rank=2,
                scope_key="scc_drift_subsystem:activity|sleep",
                reason="repair internal SCC drift subsystem",
                failure_family=("dynamics_stability",),
                construct_names=("activity", "sleep"),
            ),
        )

        sleep_prompt = next(
            block for block in repair_plan.prompt_blocks if block.id == "effects:sleep"
        )
        assert repair_plan.uses_repair_campaign is True
        assert sleep_prompt.parameter_names == ("beta_activity_sleep",)
        assert sleep_prompt.construct_names == ("activity", "sleep")
        assert "stress_vas" not in sleep_prompt.variable_names
        assert sleep_prompt.expand_neighbor_topology is False

    def test_prior_failure_classification_treats_partial_dynamics_codes_as_drift_related(self):
        _causal_spec, _skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
        _set_runtime_block(plan, runtime, "dynamics:sleep")

        validation = AssemblyValidation(
            compile_ok=True,
            pp_checked=True,
            pp_valid=False,
            diagnostics=[
                PriorValidationResult(
                    parameter="dynamics_stability",
                    is_valid=False,
                    code="partial_dynamics_budget_exhausted",
                    origin="prior_predictive",
                    issue="The sleep row has no conservative damping headroom yet.",
                    related_parameters=["rho_sleep", "sigma_sleep"],
                    failure_stage="latent_dynamics",
                    pathology_certificate=PriorPathologyCertificate(
                        kind="dynamics_stability",
                        primary_score=0.2,
                    ),
                )
            ],
        )

        repair_plan = classify_prior_failure_blocks(
            plan,
            _require_plan_block(plan, "dynamics:sleep"),
            validation,
            runtime,
        )

        assert repair_plan.scope.scope_kind == "local_drift_motif"
        assert repair_plan.block_ids == ("dynamics:sleep",)

    def test_validate_dynamics_block_partial_drift_treats_budget_overrun_as_advisory(
        self, monkeypatch
    ):
        from causal_ssm_agent.flows.stages.stage4.agentic import (
            stage4_partial_drift as partial_drift_module,
        )

        state = partial_drift_module._PartialDriftState(
            latent_names=("activity", "sleep"),
            diag_mu=np.array([0.4, 0.6]),
            diag_sigma=np.array([0.1, 0.2]),
            diag_present=np.array([True, True]),
            diag_parameter_by_index={0: "rho_activity", 1: "rho_sleep"},
            offdiag_positions=((1, 0),),
            offdiag_mu=np.array([0.5]),
            offdiag_sigma=np.array([0.3]),
            offdiag_present=np.array([True]),
            offdiag_parameter_by_index={0: "beta_activity_sleep"},
            stability_margin=0.05,
        )

        monkeypatch.setattr(
            partial_drift_module,
            "_build_partial_drift_state",
            lambda **_kwargs: state,
        )

        result = partial_drift_module.validate_dynamics_block_partial_drift(
            model_spec={"parameters": [{"name": "rho_sleep"}]},
            authored_priors={
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
            causal_spec={},
            active_construct_names=("sleep",),
            active_parameter_names=("rho_sleep", "sigma_sleep"),
        )

        assert result is None

    def test_validate_effect_block_partial_drift_treats_budget_overrun_as_advisory(
        self, monkeypatch
    ):
        from causal_ssm_agent.flows.stages.stage4.agentic import (
            stage4_partial_drift as partial_drift_module,
        )

        state = partial_drift_module._PartialDriftState(
            latent_names=("activity", "sleep"),
            diag_mu=np.array([0.4, 0.6]),
            diag_sigma=np.array([0.1, 0.2]),
            diag_present=np.array([True, True]),
            diag_parameter_by_index={0: "rho_activity", 1: "rho_sleep"},
            offdiag_positions=((1, 0),),
            offdiag_mu=np.array([0.5]),
            offdiag_sigma=np.array([0.3]),
            offdiag_present=np.array([True]),
            offdiag_parameter_by_index={0: "beta_activity_sleep"},
            stability_margin=0.05,
        )

        monkeypatch.setattr(
            partial_drift_module,
            "_build_partial_drift_state",
            lambda **_kwargs: state,
        )

        result = partial_drift_module.validate_effect_block_partial_drift(
            model_spec={"parameters": [{"name": "beta_activity_sleep"}]},
            authored_priors={
                "rho_activity": {"distribution": "Beta"},
                "rho_sleep": {"distribution": "Beta"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
            causal_spec={},
            target_construct="sleep",
            active_parameter_names=("beta_activity_sleep",),
        )

        assert result is None

    def test_compile_failure_route_prefers_true_indicator_owner_for_exact_match(self):
        from causal_ssm_agent.flows.stages.stage4.agentic.stage4_repair import (
            classify_compile_failure_route,
        )

        causal_spec, skeleton, plan, _runtime, _data_for_model = _make_stage4_mechanics_context()
        del causal_spec, skeleton

        repair_plan = classify_compile_failure_route(
            plan,
            _require_plan_block(plan, "measurement:activity"),
            "Compile error: steps has incompatible support metadata.",
        )

        assert repair_plan.scope.scope_kind == "compile_local"
        assert repair_plan.block_ids == ("indicator:steps",)
        assert repair_plan.uses_repair_campaign is False

    def test_compute_stage4_validate_step_escalates_unattributed_global_failure_to_prior_review(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
            },
        )
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "global repair trigger",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                        )
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback.startswith("PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED")
        assert "NaN/Inf detected" in feedback
        assert runtime.domain.block_status["effects:sleep"] == "accepted"
        assert runtime.domain.block_status["review:prior_system"] == "reopened"
        assert _require_active_plan_block(plan, runtime).id == "review:prior_system"
        assert get_stage4_phase(runtime, plan=plan) == "global_prior_review"
        assert "beta_activity_sleep" in runtime.domain.accepted.authored_priors

    def test_compute_stage4_validate_step_escalates_attributed_global_ppc_failure_to_prior_review(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "effects:sleep")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "late_evening_google_activity_count",
                        "distribution": "negative_binomial",
                        "link": "log",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
            },
        )
        effect_payload = {
            "block_id": "effects:sleep",
            "block_kind": "effect_prior",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "global overflow repair trigger",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue=(
                                "Predictive log-link mean overflow before observation sampling: "
                                "linear predictor exceeded the finite exp range."
                            ),
                            suggested_adjustment="Tighten the log-link priors",
                            bad_manifest_names=["late_evening_google_activity_count"],
                            first_bad_time_index=29,
                        )
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        stage_output, feedback = _apply_stage4_step_and_capture(
            effect_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            stage4_grounding_fn=stub_stage4_grounding,
        )

        assert stage_output is not None
        assert feedback.startswith("PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED")
        assert "Predictive log-link mean overflow" in feedback
        assert runtime.domain.block_status["effects:sleep"] == "accepted"
        assert runtime.domain.block_status["review:prior_system"] == "reopened"
        assert _require_active_plan_block(plan, runtime).id == "review:prior_system"
        assert get_stage4_phase(runtime, plan=plan) == "global_prior_review"
        assert runtime.domain.block_status.get("indicator:late_evening_google_activity_count") != (
            "reopened"
        )
        assert "beta_activity_sleep" in runtime.domain.accepted.authored_priors

    def test_compute_stage4_validate_step_raises_after_review_block_retry_budget_exhausted(
        self,
    ):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "review:prior_system")
        runtime.domain.block_status["review:prior_system"] = "reopened"
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
                "cor0_activity_sleep": {"distribution": "Normal"},
            },
        )
        review_payload = {
            "block_id": "review:prior_system",
            "block_kind": "global_prior_review",
            "proposal": {
                "priors": {
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.15},
                        "sources": [],
                        "reasoning": "global repair attempt",
                    }
                }
            },
        }

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return {
                "authored_priors": authored_priors,
                "validation": AssemblyValidation(
                    normalized_model_spec=current.get("model_spec"),
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="prior_predictive",
                            is_valid=False,
                            code="prior_predictive_nonfinite_samples",
                            origin="prior_predictive",
                            issue="NaN/Inf detected in sample sites: observations",
                            suggested_adjustment="Check for degenerate priors",
                            related_parameters=["drift_offdiag"],
                            supporting_codes=["dt_ct_approximation_warning"],
                        )
                    ],
                ),
            }, "PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED"

        for expected_attempt in range(1, 6):
            _apply_stage4_step_and_capture(
                review_payload,
                plan,
                runtime,
                skeleton=skeleton,
                causal_spec=causal_spec,
                stage4_grounding_fn=stub_stage4_grounding,
            )
            assert runtime.domain.repair_campaign is not None
            assert runtime.domain.repair_campaign.scope_key == "global_prior_review:prior_system"
            assert runtime.domain.repair_campaign.attempts_at_scope == expected_attempt
            assert _require_active_plan_block(plan, runtime).id == "review:prior_system"
            assert get_stage4_phase(runtime, plan=plan) == "global_prior_review"

        with pytest.raises(
            ValueError,
            match="exhausted the deterministic repair-scope ladder for a global prior-predictive failure",
        ):
            _apply_stage4_step_and_capture(
                review_payload,
                plan,
                runtime,
                skeleton=skeleton,
                causal_spec=causal_spec,
                stage4_grounding_fn=stub_stage4_grounding,
            )

    def test_global_prior_review_filters_redundant_priors_before_repair_routing(
        self,
        monkeypatch,
    ):
        from causal_ssm_agent.flows.stages.stage4.grounding import stage4_grounding

        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)

        def stub_validate_assembly(
            model_spec,
            priors,
            data_for_model,
            indicator_audits,
            causal_spec,
            *,
            skip_ppc=False,
        ):
            del priors, data_for_model, indicator_audits, causal_spec, skip_ppc
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=True,
                compiled_ssm={"compiled": True},
                pp_checked=True,
                pp_valid=False,
                diagnostics=[
                    PriorValidationResult(
                        parameter="prior_predictive",
                        is_valid=False,
                        code="prior_predictive_nonfinite_samples",
                        origin="prior_predictive",
                        issue="NaN/Inf detected in sample sites: observations",
                        suggested_adjustment="Check for degenerate priors",
                        related_parameters=["drift_offdiag"],
                        supporting_codes=["dt_ct_approximation_warning"],
                    )
                ],
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.assembly.validate_assembly",
            stub_validate_assembly,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.models.ssm_compiler.resolve_prior_proposals",
            lambda *_args, **_kwargs: [
                {"parameter": "sigma_activity"},
                {"parameter": "beta_activity_sleep"},
            ],
        )

        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "review:prior_system")
        runtime.domain.block_status["review:prior_system"] = "reopened"
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "likelihoods": [
                    {
                        "variable": "activity_vas",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                    {
                        "variable": "sleep_quality",
                        "distribution": "ordered_logistic",
                        "link": "logit",
                    },
                ],
                "parameters": [
                    {"name": "lambda_activity_vas_activity"},
                    {"name": "lambda_sleep_quality_sleep"},
                    {"name": "rho_activity"},
                    {"name": "sigma_activity"},
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                    {"name": "beta_activity_sleep"},
                    {"name": "cor0_activity_sleep"},
                ],
            },
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "lambda_sleep_quality_sleep": {"distribution": "HalfNormal"},
                "rho_activity": {"distribution": "Beta"},
                "sigma_activity": {
                    "parameter": "sigma_activity",
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.2},
                    "sources": [],
                    "reasoning": "accepted prior",
                },
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {
                    "parameter": "beta_activity_sleep",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.3},
                    "sources": [],
                    "reasoning": "accepted prior",
                },
                "cor0_activity_sleep": {"distribution": "Normal"},
            },
        )
        review_payload = {
            "block_id": "review:prior_system",
            "block_kind": "global_prior_review",
            "proposal": {
                "priors": {
                    "sigma_activity": dict(
                        runtime.domain.accepted.authored_priors["sigma_activity"]
                    ),
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.15},
                        "sources": [],
                        "reasoning": "global repair attempt",
                    },
                }
            },
        }

        _apply_stage4_step_and_capture(
            review_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            stage4_grounding_fn=stage4_grounding,
        )

        assert runtime.domain.repair_campaign is not None
        assert runtime.domain.repair_campaign.scope_key == "global_prior_review:prior_system"
        assert runtime.domain.repair_campaign.attempts_at_scope == 1
        assert runtime.interaction.last_validation_packet is not None
        assert runtime.interaction.last_validation_packet.status == "prior_predictive_failure"
        assert runtime.interaction.last_validation_packet.changed_parameters == (
            "beta_activity_sleep",
        )

        _apply_stage4_step_and_capture(
            review_payload,
            plan,
            runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            stage4_grounding_fn=stage4_grounding,
        )

        assert runtime.domain.repair_campaign is not None
        assert runtime.domain.repair_campaign.attempts_at_scope == 2

    def test_compute_stage4_validate_step_rejects_calls_after_completion(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "done"}]},
            authored_priors={
                "lambda_activity_vas_activity": {"distribution": "HalfNormal"},
                "sigma_activity": {"distribution": "HalfNormal"},
                "rho_sleep": {"distribution": "Beta"},
                "sigma_sleep": {"distribution": "HalfNormal"},
                "beta_activity_sleep": {"distribution": "Normal"},
            },
        )
        _set_done_cursor(runtime)

        stage_output, feedback = compute_stage4_validate_step(
            {},
            plan=plan,
            runtime=runtime,
            deps=_make_stage4_deps(
                causal_spec=causal_spec,
                skeleton=skeleton,
                data_for_model=data_for_model,
                stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                    "grounding should not run after completion"
                ),
            ),
        )

        assert stage_output is None
        assert feedback == "VALIDATION ERRORS:\n- no active Stage 4 frontier block remains"
        assert runtime.interaction.last_validation_packet is None

    def test_compute_stage4_validate_step_tracks_frontier_path_without_llm(self, monkeypatch):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context(
            accept_default_configuration=True
        )

        submissions = [
            {
                "block_id": "indicator:steps",
                "block_kind": "indicator_decision",
                "proposal": {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "Step counts are nonnegative integers.",
                },
            },
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The locked likelihoods and loading choices are coherent.",
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": _activity_measurement_prior_bundle(
                        lambda_sigma=0.4,
                        lambda_reasoning="initial measurement prior",
                    )
                },
            },
            {
                "block_id": "observation:manifest_mean_steps",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "manifest_mean_steps": {
                            "parameter": "manifest_mean_steps",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "steps observation intercept prior",
                        }
                    }
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_activity": {
                            "parameter": "rho_activity",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "stable activity baseline persistence",
                        },
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "stable activity residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "bad sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "paired with bad dynamics prior",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "corrected sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "corrected sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "effects:sleep",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_sleep": {
                            "parameter": "beta_activity_sleep",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "effect prior that exposes measurement mismatch",
                        }
                    }
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": _activity_loading_prior_bundle(
                        lambda_sigma=0.25,
                        lambda_reasoning="corrected measurement prior",
                    )
                },
            },
            {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_activity": {
                            "parameter": "rho_activity",
                            "distribution": "Beta",
                            "params": {"alpha": 4.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "corrected activity baseline persistence",
                        },
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "corrected activity residual scale",
                        },
                    }
                },
            },
        ]

        expected_blocks = [
            "indicator:steps",
            "review:model_spec",
            "measurement:activity",
            "observation:manifest_mean_steps",
            "observation:obs_ordered_base",
            "dynamics:activity",
            "dynamics:sleep",
            "dynamics:sleep",
            "effects:sleep",
            "measurement:activity",
            "dynamics:activity",
        ]
        expected_reopen_ids = [
            None,
            None,
            None,
            None,
            None,
            None,
            "dynamics:sleep",
            None,
            "measurement:activity",
            "dynamics:activity",
            None,
        ]

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                model_spec = data["model_spec"]
                return {
                    "model_spec": model_spec,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            priors = data["priors"]
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(priors)
            model_spec = current.get("model_spec") or runtime.domain.accepted.model_spec

            if (
                priors.get("lambda_activity_vas_activity", {}).get("reasoning")
                == "initial measurement prior"
            ):
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "manifest_mean_steps" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "obs_ordered_base" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "sigma_activity" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if priors.get("rho_sleep", {}).get("reasoning") == "bad sleep dynamics prior":
                return {
                    "model_spec": model_spec,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=False,
                        compile_error="rho_sleep interval instability",
                    ),
                }, "COMPILE ERROR:\nrho_sleep interval instability"

            if "rho_sleep" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "beta_activity_sleep" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=False,
                        diagnostics=[
                            PriorValidationResult(
                                parameter="scale_activity_vas",
                                is_valid=False,
                                code="scale_mismatch",
                                origin="prior_predictive",
                                issue="Scale mismatch for activity_vas",
                                suggested_adjustment="Tighten the measurement prior",
                            )
                        ],
                    ),
                }, "PRIOR PREDICTIVE CHECKS FAILED"

            if (
                priors.get("lambda_activity_vas_activity", {}).get("reasoning")
                == "corrected measurement prior"
            ):
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                }, "VALID"

            raise AssertionError(f"Unexpected Stage 4 grounding payload: {data}")

        def stub_validate_assembly(model_spec, authored_priors, *_args, **_kwargs):
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=True,
                pp_checked=True,
                pp_valid=True,
                diagnostics=[],
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.assembly.validate_assembly",
            stub_validate_assembly,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_reducer._finalize_repair_campaign_if_complete",
            _stub_stage4_repair_barrier_success,
        )

        visited_blocks: list[str] = []
        reopen_ids: list[str | None] = []
        for expected_block, _expected_reopen_id, payload in zip(
            expected_blocks, expected_reopen_ids, submissions, strict=True
        ):
            active_block = get_active_plan_block(plan, runtime)
            assert active_block is not None
            visited_blocks.append(active_block.id)
            assert active_block.id == expected_block

            _apply_stage4_step_and_capture(
                payload,
                plan,
                runtime,
                skeleton=skeleton,
                causal_spec=causal_spec,
                data_for_model=data_for_model,
                stage4_grounding_fn=stub_stage4_grounding,
            )
            reopened_block = get_active_plan_block(plan, runtime)
            reopen_ids.append(
                reopened_block.id
                if reopened_block is not None
                and runtime.domain.block_status.get(reopened_block.id) == "reopened"
                else None
            )

        assert visited_blocks == expected_blocks
        assert reopen_ids == expected_reopen_ids
        assert get_active_plan_block(plan, runtime) is None
        assert get_stage4_phase(runtime, plan=plan) == "done"
        assert sorted(runtime.domain.accepted.authored_priors) == [
            "beta_activity_sleep",
            "lambda_activity_vas_activity",
            "manifest_mean_steps",
            "obs_ordered_base",
            "obs_sd_activity_vas",
            "obs_sd_steps",
            "rho_activity",
            "rho_sleep",
            "sigma_activity",
            "sigma_sleep",
        ]

    def test_run_stage4_can_follow_scripted_submit_tool_path(self, monkeypatch):
        causal_spec = _make_stage4_mechanics_spec()

        submissions = [
            {
                "block_id": "indicator:steps",
                "block_kind": "indicator_decision",
                "proposal": {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "Step counts are nonnegative integers.",
                },
            },
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The locked likelihoods and loading choices are coherent.",
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": _activity_measurement_prior_bundle(
                        lambda_sigma=0.25,
                        lambda_reasoning="measurement prior",
                    )
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_activity": {
                            "parameter": "rho_activity",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "activity baseline persistence",
                        },
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "activity residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "effects:sleep",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_sleep": {
                            "parameter": "beta_activity_sleep",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "effect prior",
                        }
                    }
                },
            },
        ]
        visited_blocks: list[str] = []
        visible_tools: list[list[str]] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                model_spec = data["model_spec"]
                return _make_stub_grounding_result(
                    {
                        "model_spec": model_spec,
                        "validation": AssemblyValidation(
                            normalized_model_spec=model_spec,
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked="beta_activity_sleep" in authored_priors,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_reducer._finalize_repair_campaign_if_complete",
            _stub_stage4_repair_barrier_success,
        )

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="Does activity improve sleep?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                session_factory=_make_stage4_session_factory(
                    _make_scripted_stage4_generate(
                        submissions,
                        visited_blocks=visited_blocks,
                        visible_tools=visible_tools,
                    )
                ),
                enable_literature=False,
                enable_paraphrasing=False,
            )
        )

        assert visited_blocks == [
            "indicator:steps",
            "review:model_spec",
            "measurement:activity",
            "observation:obs_ordered_base",
            "dynamics:activity",
            "dynamics:sleep",
            "effects:sleep",
        ]
        assert visible_tools == [
            ["submit_indicator_choice"],
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert any(
            likelihood["variable"] == "steps" and likelihood["distribution"] == "poisson"
            for likelihood in result.model_spec["likelihoods"]
        )
        assert sorted(result.authored_priors) == [
            "beta_activity_sleep",
            "lambda_activity_vas_activity",
            "manifest_mean_steps",
            "obs_ordered_base",
            "obs_sd_activity_vas",
            "obs_sd_steps",
            "rho_activity",
            "rho_sleep",
            "sigma_activity",
            "sigma_sleep",
        ]

    def test_run_stage4_effect_blocks_run_sequentially_to_final_validation(self, monkeypatch):
        causal_spec = _make_stage4_two_effect_spec()

        submissions_by_block = {
            "indicator:steps": {
                "block_id": "indicator:steps",
                "block_kind": "indicator_decision",
                "proposal": {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "Step counts are nonnegative integers.",
                },
            },
            "review:model_spec": {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The locked likelihoods and loading choices are coherent.",
                },
            },
            "measurement:activity": {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": _activity_measurement_prior_bundle(
                        lambda_sigma=0.25,
                        lambda_reasoning="measurement prior",
                    )
                },
            },
            "observation:obs_ordered_base": {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            "dynamics:activity": {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_activity": {
                            "parameter": "rho_activity",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "activity baseline persistence",
                        },
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.4},
                            "sources": [],
                            "reasoning": "activity residual scale",
                        },
                    }
                },
            },
            "dynamics:sleep": {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep persistence",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
            "dynamics:mood": {
                "block_id": "dynamics:mood",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_mood": {
                            "parameter": "rho_mood",
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "mood persistence",
                        },
                        "sigma_mood": {
                            "parameter": "sigma_mood",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "mood residual scale",
                        },
                    }
                },
            },
            "effects:sleep": {
                "block_id": "effects:sleep",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_sleep": {
                            "parameter": "beta_activity_sleep",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "sleep effect prior",
                        }
                    }
                },
            },
            "effects:mood": {
                "block_id": "effects:mood",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_mood": {
                            "parameter": "beta_activity_mood",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "mood effect prior",
                        }
                    }
                },
            },
        }
        visited_blocks: list[str] = []
        visible_tools: list[list[str]] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                model_spec = data["model_spec"]
                return _make_stub_grounding_result(
                    {
                        "model_spec": model_spec,
                        "validation": AssemblyValidation(
                            normalized_model_spec=model_spec,
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            pp_checked = {"beta_activity_sleep", "beta_activity_mood"}.issubset(authored_priors)
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=pp_checked,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        def stub_validate_assembly(model_spec, authored_priors, *_args, **_kwargs):
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=True,
                pp_checked=True,
                pp_valid=True,
                diagnostics=[],
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.assembly.validate_assembly",
            stub_validate_assembly,
        )

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="Does activity affect sleep and mood?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                session_factory=_make_stage4_session_factory(
                    _make_scripted_stage4_generate_by_block(
                        submissions_by_block,
                        visited_blocks=visited_blocks,
                        visible_tools=visible_tools,
                    )
                ),
                enable_literature=False,
                enable_paraphrasing=False,
            )
        )

        assert visited_blocks == [
            "indicator:steps",
            "review:model_spec",
            "measurement:activity",
            "observation:obs_ordered_base",
            "dynamics:activity",
            "dynamics:sleep",
            "dynamics:mood",
            "effects:sleep",
            "effects:mood",
        ]
        assert visible_tools == [
            ["submit_indicator_choice"],
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert sorted(result.authored_priors) == [
            "beta_activity_mood",
            "beta_activity_sleep",
            "lambda_activity_vas_activity",
            "manifest_mean_steps",
            "obs_ordered_base",
            "obs_sd_activity_vas",
            "obs_sd_steps",
            "rho_activity",
            "rho_mood",
            "rho_sleep",
            "sigma_activity",
            "sigma_mood",
            "sigma_sleep",
        ]
        assert result.validation is not None
        assert result.validation.pp_checked is True
        assert result.validation.pp_valid is True

    def test_run_stage4_auto_locks_initial_model_spec_when_no_model_blocks(self, monkeypatch):
        causal_spec = _make_stage4_no_model_block_spec()
        submissions = [
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The deterministic model form is coherent.",
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
        ]
        visited_blocks: list[str] = []
        visible_tools: list[list[str]] = []
        model_spec_calls: list[dict] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                model_spec_calls.append(data["model_spec"])
                return _make_stub_grounding_result(
                    {
                        "model_spec": data["model_spec"],
                        "validation": AssemblyValidation(
                            normalized_model_spec=data["model_spec"],
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_reducer._finalize_repair_campaign_if_complete",
            _stub_stage4_repair_barrier_success,
        )

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="How persistent is sleep quality?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                session_factory=_make_stage4_session_factory(
                    _make_scripted_stage4_generate(
                        submissions,
                        visited_blocks=visited_blocks,
                        visible_tools=visible_tools,
                    )
                ),
                enable_literature=False,
                enable_paraphrasing=False,
            )
        )

        assert len(model_spec_calls) == 1
        assert visited_blocks == [
            "review:model_spec",
            "observation:obs_ordered_base",
            "dynamics:sleep",
        ]
        assert visible_tools == [
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert sorted(result.authored_priors) == ["obs_ordered_base", "rho_sleep", "sigma_sleep"]

    def test_run_stage4_tracks_submission_when_feedback_repeats(self, monkeypatch):
        causal_spec = _make_stage4_no_model_block_spec()
        submissions = [
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The deterministic model form is coherent.",
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep dynamics prior",
                        },
                    }
                },
            },
        ]
        visited_blocks: list[str] = []
        visible_tools: list[list[str]] = []
        prior_attempts = 0

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            nonlocal prior_attempts
            current = current or {}
            if "model_spec" in data:
                return _make_stub_grounding_result(
                    {
                        "model_spec": data["model_spec"],
                        "validation": AssemblyValidation(
                            normalized_model_spec=data["model_spec"],
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            if "obs_ordered_base" in data["priors"]:
                return _make_stub_grounding_result(
                    {
                        "authored_priors": authored_priors,
                        "validation": AssemblyValidation(
                            normalized_model_spec=current.get("model_spec"),
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            prior_attempts += 1
            if prior_attempts < 3:
                return _make_stub_grounding_result(
                    {
                        "authored_priors": authored_priors,
                        "validation": AssemblyValidation(
                            normalized_model_spec=current.get("model_spec"),
                            compile_ok=True,
                            pp_checked=True,
                            pp_valid=False,
                            diagnostics=[
                                PriorValidationResult(
                                    parameter="rho_sleep",
                                    is_valid=False,
                                    code=f"local_prior_adjustment_{prior_attempts}",
                                    origin="prior_predictive",
                                    issue="Sleep persistence prior still needs adjustment",
                                    suggested_adjustment="Tighten the active dynamics prior",
                                )
                            ],
                        ),
                    },
                    "PRIOR PREDICTIVE FEEDBACK:\n- still failing",
                )

            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_reducer._finalize_repair_campaign_if_complete",
            _stub_stage4_repair_barrier_success,
        )

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="How persistent is sleep quality?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                session_factory=_make_stage4_session_factory(
                    _make_scripted_stage4_generate(
                        submissions,
                        visited_blocks=visited_blocks,
                        visible_tools=visible_tools,
                    )
                ),
                enable_literature=False,
                enable_paraphrasing=False,
            )
        )

        assert prior_attempts == 3
        assert visited_blocks == [
            "review:model_spec",
            "observation:obs_ordered_base",
            "dynamics:sleep",
            "dynamics:sleep",
            "dynamics:sleep",
        ]
        assert visible_tools == [
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert sorted(result.authored_priors) == ["obs_ordered_base", "rho_sleep"]

    def test_run_stage4_resumes_from_runtime_checkpoint(self, monkeypatch):
        causal_spec = _make_stage4_no_model_block_spec()
        review_submission = {
            "block_id": "review:model_spec",
            "block_kind": "global_review",
            "proposal": {
                "decision": "approve",
                "reasoning": "The deterministic model form is coherent.",
            },
        }
        observation_submission = {
            "block_id": "observation:obs_ordered_base",
            "block_kind": "observation_prior",
            "proposal": {
                "priors": {
                    "obs_ordered_base": {
                        "parameter": "obs_ordered_base",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "sources": [],
                        "reasoning": "ordered-threshold location prior",
                    }
                }
            },
        }
        dynamics_submission = {
            "block_id": "dynamics:sleep",
            "block_kind": "dynamics_prior",
            "proposal": {
                "priors": {
                    "rho_sleep": {
                        "parameter": "rho_sleep",
                        "distribution": "Beta",
                        "params": {"alpha": 3.0, "beta": 2.0},
                        "sources": [],
                        "reasoning": "sleep persistence",
                    },
                    "sigma_sleep": {
                        "parameter": "sigma_sleep",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.35},
                        "sources": [],
                        "reasoning": "sleep residual scale",
                    },
                }
            },
        }
        saved_runtime: Stage4Runtime | None = None
        first_run_blocks: list[str] = []
        second_run_blocks: list[str] = []
        clear_calls = 0

        def save_checkpoint(runtime: Stage4Runtime) -> None:
            nonlocal saved_runtime
            saved_runtime = deepcopy(runtime)

        def clear_checkpoint() -> None:
            nonlocal clear_calls, saved_runtime
            clear_calls += 1
            saved_runtime = None

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                return _make_stub_grounding_result(
                    {
                        "model_spec": data["model_spec"],
                        "validation": AssemblyValidation(
                            normalized_model_spec=data["model_spec"],
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )

        async def fail_after_review(
            messages,
            tools,
            rewrite_messages=None,
            rewrite_tools=None,
            label=None,
        ):
            del messages, rewrite_messages, rewrite_tools
            assert label is not None
            assert label.startswith("stage-4:")
            block_id = label.removeprefix("stage-4:")
            if block_id == "model:configuration":
                submit_tool = next(
                    tool for tool in tools if tool.name == "submit_model_configuration"
                )
                feedback = await submit_tool(
                    initialization_policy="stationary",
                    observation_intercept_policy="free",
                    equilibrium_forcing=False,
                    reasoning="Default Stage 4 test configuration.",
                )
                assert isinstance(feedback, str)
                assert not feedback.startswith("VALIDATION ERRORS:")
                return ""
            first_run_blocks.append(block_id)
            if block_id == "review:model_spec":
                submit_tool = next(tool for tool in tools if tool.name == "submit_model_review")
                feedback = await submit_tool(**_stage4_submit_tool_args(review_submission))
                assert isinstance(feedback, str)
                assert not feedback.startswith("VALIDATION ERRORS:")
                return ""
            raise RuntimeError("OpenRouter returned no choices")

        with pytest.raises(RuntimeError, match="OpenRouter returned no choices"):
            asyncio.run(
                run_stage4(
                    causal_spec=causal_spec,
                    question="How persistent is sleep quality?",
                    data_for_model=pl.DataFrame(),
                    indicator_audits={},
                    session_factory=_make_stage4_session_factory(fail_after_review),
                    enable_literature=False,
                    enable_paraphrasing=False,
                    save_checkpoint=save_checkpoint,
                )
            )

        assert saved_runtime is not None
        assert first_run_blocks == ["review:model_spec", "observation:obs_ordered_base"]

        async def resume_from_dynamics(
            messages,
            tools,
            rewrite_messages=None,
            rewrite_tools=None,
            label=None,
        ):
            del messages, rewrite_messages, rewrite_tools
            assert label is not None
            assert label.startswith("stage-4:")
            block_id = label.removeprefix("stage-4:")
            second_run_blocks.append(block_id)
            submission = (
                observation_submission
                if block_id == "observation:obs_ordered_base"
                else dynamics_submission
            )
            submit_tool = next(
                tool
                for tool in tools
                if tool.name == _stage4_submit_tool_name(submission["block_kind"])
            )
            feedback = await submit_tool(**_stage4_submit_tool_args(submission))
            assert isinstance(feedback, str)
            assert not feedback.startswith("VALIDATION ERRORS:")
            return ""

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="How persistent is sleep quality?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                session_factory=_make_stage4_session_factory(resume_from_dynamics),
                enable_literature=False,
                enable_paraphrasing=False,
                load_checkpoint=lambda: saved_runtime,
                save_checkpoint=save_checkpoint,
                clear_checkpoint=clear_checkpoint,
            )
        )

        assert second_run_blocks == ["observation:obs_ordered_base", "dynamics:sleep"]
        assert clear_calls == 0
        assert sorted(result.authored_priors) == ["obs_ordered_base", "rho_sleep", "sigma_sleep"]

    def test_load_resumable_stage4_runtime_resets_run_local_repair_retry_state(self):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "review:prior_system")
        runtime.domain.block_status["review:prior_system"] = "reopened"
        runtime.domain.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(("prior_predictive_observation_mean_overflow",),),
            scope_kind="global_prior_review",
            scope_key="global_prior_review:prior_system",
            scope_rank=3,
            scope_block_ids=("review:prior_system",),
            prompt_blocks_by_id={
                "review:prior_system": _require_plan_block(plan, "review:prior_system"),
            },
            completed_block_ids=frozenset(),
            attempts_at_scope=2,
            best_certificate=PriorPathologyCertificate(
                kind="nonfinite_samples",
                primary_score=0.5,
                secondary_score=8.0,
            ),
        )

        resumed = _load_resumable_stage4_runtime(
            plan,
            load_checkpoint=lambda: runtime,
            clear_checkpoint=lambda: pytest.fail("compatible checkpoint should not be cleared"),
        )

        assert resumed is runtime
        assert resumed.domain.repair_campaign is not None
        assert resumed.domain.repair_campaign.attempts_at_scope == 1
        assert resumed.domain.repair_campaign.best_certificate is None
        assert resumed.domain.active_block_id == "review:prior_system"

    def test_run_stage4_aborts_immediately_on_fatal_submit_failure(self, monkeypatch):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        loaded_runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, loaded_runtime, "review:prior_system")
        loaded_runtime.domain.block_status["review:prior_system"] = "reopened"
        loaded_runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "beta_activity_sleep"}]},
            authored_priors={
                "beta_activity_sleep": {
                    "parameter": "beta_activity_sleep",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.2},
                    "sources": [],
                    "reasoning": "accepted prior",
                }
            },
        )
        visited_blocks: list[str] = []
        save_calls: list[Stage4Runtime] = []

        def fail_compute(payload, *, plan, runtime, deps):
            del payload, plan, deps
            runtime.domain.active_block_id = None
            runtime.domain.block_status["review:prior_system"] = "accepted"
            raise ValueError("Stage 4 exhausted the deterministic repair-scope ladder")

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_session.compute_stage4_validate_step_with_transitions",
            fail_compute,
        )

        async def fatal_generate(
            messages,
            tools,
            rewrite_messages=None,
            rewrite_tools=None,
            label=None,
        ):
            del messages, rewrite_messages, rewrite_tools
            assert label is not None
            assert label.startswith("stage-4:")
            block_id = label.removeprefix("stage-4:")
            visited_blocks.append(block_id)
            submit_tool = next(tool for tool in tools if tool.name == "submit_prior_block")
            await submit_tool(
                priors={
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.1},
                        "sources": [],
                        "reasoning": "repair attempt",
                    }
                }
            )
            raise AssertionError("fatal submit should have aborted the run")

        with pytest.raises(
            Stage4FatalSubmissionError,
            match="Stage 4 reducer failed while applying a submit tool",
        ):
            asyncio.run(
                run_stage4(
                    causal_spec=causal_spec,
                    question="How should the prior system repair proceed?",
                    data_for_model=pl.DataFrame(),
                    indicator_audits={},
                    session_factory=_make_stage4_session_factory(fatal_generate),
                    enable_literature=False,
                    enable_paraphrasing=False,
                    load_checkpoint=lambda: loaded_runtime,
                    save_checkpoint=lambda runtime: save_calls.append(deepcopy(runtime)),
                )
            )

        assert visited_blocks == ["review:prior_system"]
        assert save_calls == []
        assert loaded_runtime.domain.active_block_id == "review:prior_system"
        assert loaded_runtime.domain.block_status["review:prior_system"] == "reopened"

    def test_run_stage4_discards_invalid_runtime_checkpoint(self, monkeypatch):
        causal_spec = _make_stage4_no_model_block_spec()
        visited_blocks: list[str] = []
        cleared: list[bool] = []

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = _current_stage4_state(current)
            if "model_spec" in data:
                return _make_stub_grounding_result(
                    {
                        "model_spec": data["model_spec"],
                        "validation": AssemblyValidation(
                            normalized_model_spec=data["model_spec"],
                            compile_ok=True,
                        ),
                    },
                    "MODEL STATE SAVED:\n- missing priors",
                )

            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(data["priors"])
            return _make_stub_grounding_result(
                {
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=current.get("model_spec"),
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                },
                "VALID",
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.grounding.stage4_grounding",
            stub_stage4_grounding,
        )

        submissions_by_block = {
            "review:model_spec": {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The deterministic model form is coherent.",
                },
            },
            "observation:obs_ordered_base": {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            "dynamics:sleep": {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "sleep persistence",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "sleep residual scale",
                        },
                    }
                },
            },
        }

        result = asyncio.run(
            run_stage4(
                causal_spec=causal_spec,
                question="How persistent is sleep quality?",
                data_for_model=pl.DataFrame(),
                indicator_audits={},
                session_factory=_make_stage4_session_factory(
                    _make_scripted_stage4_generate_by_block(
                        submissions_by_block,
                        visited_blocks=visited_blocks,
                        visible_tools=[],
                    )
                ),
                enable_literature=False,
                enable_paraphrasing=False,
                load_checkpoint=lambda: {"not": "a runtime"},
                clear_checkpoint=lambda: cleared.append(True),
            )
        )

        assert cleared == [True]
        assert visited_blocks == [
            "review:model_spec",
            "observation:obs_ordered_base",
            "dynamics:sleep",
        ]
        assert sorted(result.authored_priors) == ["obs_ordered_base", "rho_sleep", "sigma_sleep"]

    def test_validate_stage4_runtime_checkpoint_rejects_stale_direct_writer_prompt_block(self):
        causal_spec, skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
        del causal_spec, skeleton, _data_for_model

        _set_runtime_block(plan, runtime, "dynamics:sleep")
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={
                "parameters": [
                    {"name": "rho_sleep"},
                    {"name": "sigma_sleep"},
                ]
            }
        )
        runtime.domain.repair_campaign = Stage4RepairCampaignState(
            failure_family_key=(("prior_predictive_nonfinite_samples",),),
            scope_kind="direct_writer_blocks",
            scope_key="direct_writer_blocks:rho_sleep",
            scope_rank=0,
            scope_block_ids=("dynamics:sleep",),
            prompt_blocks_by_id={
                "dynamics:sleep": _require_plan_block(plan, "dynamics:sleep"),
            },
        )

        incompatibility = _validate_stage4_runtime_checkpoint(plan, runtime)

        assert (
            incompatibility
            == "checkpoint direct-writer prompt blocks no longer match the scoped repair surface"
        )

    def test_validate_stage4_runtime_checkpoint_rejects_done_state_with_invalid_validation(self):
        causal_spec, skeleton, plan, runtime, _data_for_model = _make_stage4_mechanics_context()
        del causal_spec, skeleton, _data_for_model

        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "rho_sleep"}]},
            authored_priors={"rho_sleep": {"distribution": "Beta"}},
            validation=AssemblyValidation(compile_ok=True, pp_checked=True, pp_valid=False),
        )
        _set_done_cursor(runtime)

        incompatibility = _validate_stage4_runtime_checkpoint(plan, runtime)

        assert (
            incompatibility
            == "checkpoint marks Stage 4 done without a valid accepted validation result"
        )

    def test_stage4_session_done_requires_valid_accepted_validation(self):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "rho_sleep"}]},
            authored_priors={"rho_sleep": {"distribution": "Beta"}},
            validation=AssemblyValidation(compile_ok=True, pp_checked=True, pp_valid=False),
        )
        _set_done_cursor(runtime)
        session = _make_stage4_session(
            question="test",
            plan=plan,
            runtime=runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                "grounding should not run for completion checks"
            ),
        )

        assert session.is_done() is False
        with pytest.raises(
            ValueError,
            match="Stage 4 session has not completed a valid model_spec \\+ priors",
        ):
            session.result()

    def test_stage4_session_submit_rolls_back_on_fatal_reducer_exception(self, monkeypatch):
        causal_spec = _make_stage4_global_repair_spec()
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _set_runtime_block(plan, runtime, "review:prior_system")
        runtime.domain.block_status["review:prior_system"] = "reopened"
        runtime.domain.accepted = Stage4AcceptedArtifacts(
            model_spec={"parameters": [{"name": "beta_activity_sleep"}]},
            authored_priors={
                "beta_activity_sleep": {
                    "parameter": "beta_activity_sleep",
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.2},
                    "sources": [],
                    "reasoning": "accepted prior",
                }
            },
        )
        session = _make_stage4_session(
            question="test",
            plan=plan,
            runtime=runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                "grounding should not run when the reducer path is patched"
            ),
        )
        persist_calls: list[tuple[Stage4Runtime, tuple[dict[str, Any], ...]]] = []
        session.persist_runtime = lambda runtime, transitions: persist_calls.append(
            (deepcopy(runtime), transitions)
        )
        snapshot = deepcopy(session.runtime)

        def fail_compute(payload, *, plan, runtime, deps):
            del payload, plan, deps
            runtime.domain.active_block_id = None
            runtime.domain.block_status["review:prior_system"] = "accepted"
            runtime.domain.accepted.authored_priors["beta_activity_sleep"] = {
                "distribution": "HalfNormal"
            }
            raise ValueError("Stage 4 exhausted the deterministic repair-scope ladder")

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_session.compute_stage4_validate_step_with_transitions",
            fail_compute,
        )

        with pytest.raises(
            Stage4FatalSubmissionError,
            match="Stage 4 reducer failed while applying a submit tool",
        ):
            session.submit_prior_block(
                priors={
                    "beta_activity_sleep": {
                        "parameter": "beta_activity_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.1},
                        "sources": [],
                        "reasoning": "repair attempt",
                    }
                }
            )

        assert persist_calls == []
        assert session.runtime == snapshot
        assert session.current_block() is not None
        assert session.current_block().id == "review:prior_system"

    def test_execute_tools_reraises_stage4_fatal_submission_error(self):
        async def _raise_fatal(*, priors):
            del priors
            raise Stage4FatalSubmissionError("fatal reducer failure")

        assistant_message = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "submit_prior_block", "arguments": '{"priors": {}}'},
                }
            ]
        }
        tool = Tool(
            name="submit_prior_block",
            description="test",
            parameters={"type": "object", "properties": {}, "required": []},
            execute=_raise_fatal,
        )

        with pytest.raises(Stage4FatalSubmissionError, match="fatal reducer failure"):
            asyncio.run(execute_tools(assistant_message, [tool]))

    def test_stage4_session_runs_model_lock_callback_after_submit(self, monkeypatch):
        causal_spec, skeleton, plan, runtime, data_for_model = _make_stage4_mechanics_context()
        session = _make_stage4_session(
            question="test",
            plan=plan,
            runtime=runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            data_for_model=data_for_model,
            stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail(
                "grounding should not run when the reducer path is patched"
            ),
        )
        persisted: list[tuple[dict[str, Any], ...]] = []
        locked_model_specs: list[dict[str, Any]] = []
        session.persist_runtime = lambda _runtime, transitions: persisted.append(transitions)
        session.on_model_spec_locked = lambda rt: locked_model_specs.append(
            deepcopy(rt.domain.accepted.model_spec)
        )

        def _lock_after_submit(payload, *, plan, runtime, deps):
            del payload, plan, deps
            runtime.domain.accepted.model_spec = {
                "parameters": [{"name": "rho_sleep"}],
                "likelihoods": [],
            }
            runtime.domain.model_lock_pending = False
            return None, "LOCKED", ({"block_id": "indicator:steps", "status": "accepted"},)

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_session.compute_stage4_validate_step_with_transitions",
            _lock_after_submit,
        )

        feedback = session.submit_indicator_choice(
            variable="steps",
            distribution="poisson",
            link="log",
            reasoning="Count data.",
        )

        assert feedback == "LOCKED"
        assert persisted == [({"block_id": "indicator:steps", "status": "accepted"},)]
        assert locked_model_specs == [{"parameters": [{"name": "rho_sleep"}], "likelihoods": []}]

    def test_stage4_tool_loop_compacts_context_while_trace_grows(self, monkeypatch):
        from causal_ssm_agent.flows.stages.stage4.tools import (
            make_submit_indicator_choice_tool,
            make_submit_model_review_tool,
            make_submit_prior_block_tool,
        )

        causal_spec = _make_stage4_mechanics_spec()
        submissions = [
            {
                "block_id": "indicator:steps",
                "block_kind": "indicator_decision",
                "proposal": {
                    "variable": "steps",
                    "distribution": "poisson",
                    "link": "log",
                    "reasoning": "Step counts are nonnegative integers.",
                },
            },
            {
                "block_id": "review:model_spec",
                "block_kind": "global_review",
                "proposal": {
                    "decision": "approve",
                    "reasoning": "The locked likelihoods and loading choices are coherent.",
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": _activity_measurement_prior_bundle(
                        lambda_sigma=0.4,
                        lambda_reasoning="initial measurement prior",
                    )
                },
            },
            {
                "block_id": "observation:manifest_mean_steps",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "manifest_mean_steps": {
                            "parameter": "manifest_mean_steps",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "steps observation intercept prior",
                        }
                    }
                },
            },
            {
                "block_id": "observation:obs_ordered_base",
                "block_kind": "observation_prior",
                "proposal": {
                    "priors": {
                        "obs_ordered_base": {
                            "parameter": "obs_ordered_base",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 1.0},
                            "sources": [],
                            "reasoning": "ordered-threshold location prior",
                        }
                    }
                },
            },
            {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_activity": {
                            "parameter": "rho_activity",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "stable activity baseline persistence",
                        },
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "stable activity residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 2.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "bad sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.5},
                            "sources": [],
                            "reasoning": "paired with bad dynamics prior",
                        },
                    }
                },
            },
            {
                "block_id": "dynamics:sleep",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_sleep": {
                            "parameter": "rho_sleep",
                            "distribution": "Beta",
                            "params": {"alpha": 3.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "corrected sleep dynamics prior",
                        },
                        "sigma_sleep": {
                            "parameter": "sigma_sleep",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "corrected sleep residual scale",
                        },
                    }
                },
            },
            {
                "block_id": "effects:sleep",
                "block_kind": "effect_prior",
                "proposal": {
                    "priors": {
                        "beta_activity_sleep": {
                            "parameter": "beta_activity_sleep",
                            "distribution": "Normal",
                            "params": {"mu": 0.0, "sigma": 0.2},
                            "sources": [],
                            "reasoning": "effect prior that exposes measurement mismatch",
                        }
                    }
                },
            },
            {
                "block_id": "measurement:activity",
                "block_kind": "measurement_prior",
                "proposal": {
                    "priors": _activity_loading_prior_bundle(
                        lambda_sigma=0.25,
                        lambda_reasoning="corrected measurement prior",
                    )
                },
            },
            {
                "block_id": "dynamics:activity",
                "block_kind": "dynamics_prior",
                "proposal": {
                    "priors": {
                        "rho_activity": {
                            "parameter": "rho_activity",
                            "distribution": "Beta",
                            "params": {"alpha": 4.0, "beta": 2.0},
                            "sources": [],
                            "reasoning": "corrected activity baseline persistence",
                        },
                        "sigma_activity": {
                            "parameter": "sigma_activity",
                            "distribution": "HalfNormal",
                            "params": {"sigma": 0.35},
                            "sources": [],
                            "reasoning": "corrected activity residual scale",
                        },
                    }
                },
            },
        ]
        expected_blocks = [
            "indicator:steps",
            "review:model_spec",
            "measurement:activity",
            "observation:manifest_mean_steps",
            "observation:obs_ordered_base",
            "dynamics:activity",
            "dynamics:sleep",
            "dynamics:sleep",
            "effects:sleep",
            "measurement:activity",
            "dynamics:activity",
        ]
        seen_block_ids: list[str] = []
        seen_feedbacks: list[str] = []
        seen_message_counts: list[int] = []
        seen_message_roles: list[list[str]] = []
        seen_tool_names: list[list[str]] = []
        trace_capture: dict[str, object] = {}
        call_index = 0
        skeleton = derive_deterministic_spec(causal_spec)
        plan = build_stage4_plan(causal_spec, skeleton)
        runtime = make_stage4_runtime(plan)
        _accept_default_model_configuration(
            causal_spec=causal_spec,
            skeleton=skeleton,
            plan=plan,
            runtime=runtime,
        )

        def stub_stage4_grounding(data, _causal_spec, current=None, **_kwargs):
            current = current or {}
            if "model_spec" in data:
                model_spec = data["model_spec"]
                return {
                    "model_spec": model_spec,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            priors = data["priors"]
            authored_priors = dict(current.get("authored_priors") or {})
            authored_priors.update(priors)
            model_spec = current.get("model_spec") or runtime.domain.accepted.model_spec

            if (
                priors.get("lambda_activity_vas_activity", {}).get("reasoning")
                == "initial measurement prior"
            ):
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "manifest_mean_steps" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "obs_ordered_base" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "sigma_activity" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if priors.get("rho_sleep", {}).get("reasoning") == "bad sleep dynamics prior":
                return {
                    "model_spec": model_spec,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=False,
                        compile_error="rho_sleep interval instability",
                    ),
                }, "COMPILE ERROR:\nrho_sleep interval instability"

            if "rho_sleep" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                    ),
                }, "MODEL STATE SAVED:\n- missing priors"

            if "beta_activity_sleep" in priors:
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=False,
                        diagnostics=[
                            PriorValidationResult(
                                parameter="scale_activity_vas",
                                is_valid=False,
                                code="scale_mismatch",
                                origin="prior_predictive",
                                issue="Scale mismatch for activity_vas",
                                suggested_adjustment="Tighten the measurement prior",
                            )
                        ],
                    ),
                }, "PRIOR PREDICTIVE CHECKS FAILED"

            if (
                priors.get("lambda_activity_vas_activity", {}).get("reasoning")
                == "corrected measurement prior"
            ):
                return {
                    "model_spec": model_spec,
                    "authored_priors": authored_priors,
                    "validation": AssemblyValidation(
                        normalized_model_spec=model_spec,
                        compile_ok=True,
                        pp_checked=True,
                        pp_valid=True,
                    ),
                }, "VALID"

            raise AssertionError(f"Unexpected Stage 4 grounding payload: {data}")

        def stub_validate_assembly(model_spec, authored_priors, *_args, **_kwargs):
            return AssemblyValidation(
                normalized_model_spec=model_spec,
                compile_ok=True,
                pp_checked=True,
                pp_valid=True,
                diagnostics=[],
            )

        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.assembly.validate_assembly",
            stub_validate_assembly,
        )
        monkeypatch.setattr(
            "causal_ssm_agent.flows.stages.stage4.agentic.stage4_reducer._finalize_repair_campaign_if_complete",
            _stub_stage4_repair_barrier_success,
        )

        session = _make_stage4_session(
            question="Does activity improve sleep?",
            plan=plan,
            runtime=runtime,
            skeleton=skeleton,
            causal_spec=causal_spec,
            stage4_grounding_fn=stub_stage4_grounding,
            model_topology={},
            loading_params=skeleton.loading_params,
            prior_cards=build_prior_cards(causal_spec, skeleton),
        )

        async def fake_call_model(model_name, messages, tools=None, config=None, log_label=None):
            nonlocal call_index
            assert model_name == "test-model"
            assert tools is not None
            turn = session.current_turn()
            assert turn is not None
            seen_message_counts.append(len(messages))
            seen_message_roles.append([message["role"] for message in messages])
            seen_tool_names.append([tool.name for tool in tools])
            seen_block_ids.append(turn.block.id)
            seen_feedbacks.append(turn.latest_feedback)
            payload = submissions[call_index]
            call_index += 1
            tool_name = _stage4_submit_tool_name(str(payload["block_kind"]))
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": f"call_{call_index}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(_stage4_submit_tool_args(payload)),
                            },
                        }
                    ],
                },
                "completion": "",
                "usage": None,
                "model": "test-model",
                "time": 0.1,
                "stop_reason": "tool_calls",
            }

        tool_map = {
            "submit_indicator_choice": make_submit_indicator_choice_tool(session),
            "submit_model_review": make_submit_model_review_tool(session),
            "submit_prior_block": make_submit_prior_block_tool(session),
        }
        monkeypatch.setattr("causal_ssm_agent.utils.llm.call_model", fake_call_model)

        generate = make_generate_fn(
            "test-model",
            config=GenerateConfig(),
            trace_capture=trace_capture,
        )
        completion = ""
        while not session.is_done():
            turn = session.current_turn()
            assert turn is not None
            completion = asyncio.run(
                _await_string(
                    generate(
                        turn.messages,
                        [tool_map[turn.required_submission_tool_name]],
                    )
                )
            )

        assert completion == ""
        assert seen_block_ids == expected_blocks
        assert call_index == len(submissions)
        assert seen_message_counts == [2] * len(submissions)
        assert seen_message_roles == [["system", "user"]] * len(submissions)
        assert seen_tool_names == [
            ["submit_indicator_choice"],
            ["submit_model_review"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
            ["submit_prior_block"],
        ]
        assert seen_feedbacks[:9] == [
            "No validator feedback yet. Submit the active block only.",
            "MODEL STATE SAVED:\n- missing priors",
            "BLOCK ACCEPTED:\n- saved `review:model_spec`\n- next block: `measurement:activity` (measurement_prior)",
            "MODEL STATE SAVED:\n- missing priors",
            "MODEL STATE SAVED:\n- missing priors",
            "MODEL STATE SAVED:\n- missing priors",
            "MODEL STATE SAVED:\n- missing priors",
            "COMPILE ERROR:\nrho_sleep interval instability",
            "MODEL STATE SAVED:\n- missing priors",
        ]
        assert seen_feedbacks[9].startswith("REPAIR CAMPAIGN ACTIVE:")
        assert "next repair block: `measurement:activity`" in seen_feedbacks[9]
        assert seen_feedbacks[10].startswith("REPAIR CAMPAIGN PROGRESS:")
        assert "next repair block: `dynamics:activity`" in seen_feedbacks[10]

        trace = _require_trace(trace_capture)
        assert len(trace.messages) == 4 * len(submissions)
        assert [message.role for message in trace.messages[:2]] == ["system", "user"]
        assert sum(message.role == "assistant" for message in trace.messages) == len(submissions)
        assert sum(message.role == "tool" for message in trace.messages) == len(submissions)
        assert len(trace.messages) > max(seen_message_counts)
        assert any(
            message.tool_result == "COMPILE ERROR:\nrho_sleep interval instability"
            for message in trace.messages
        )
        assert any(
            isinstance(message.tool_result, str)
            and message.tool_result.startswith("REPAIR CAMPAIGN ACTIVE:")
            for message in trace.messages
        )
        assert runtime.domain.accepted.model_spec is not None
        assert sorted(runtime.domain.accepted.authored_priors) == [
            "beta_activity_sleep",
            "lambda_activity_vas_activity",
            "manifest_mean_steps",
            "obs_ordered_base",
            "obs_sd_activity_vas",
            "obs_sd_steps",
            "rho_activity",
            "rho_sleep",
            "sigma_activity",
            "sigma_sleep",
        ]
