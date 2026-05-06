"""Stage 4 prompt-context rendering tests."""

from tests.stages.stage4._support import (
    AssemblyValidation,
    PriorValidationResult,
    SimpleNamespace,
    Stage4AcceptedArtifacts,
    Stage4FrontierBlock,
    Stage4Messages,
    Stage4Skeleton,
    _make_plan,
    _make_runtime,
    _make_stage4_session,
    _set_runtime_block,
    get_stage4_block_handler,
    make_stage4_validation_packet,
    pytest,
)

# --- Prompt assembly tests ---


class TestStage4Messages:
    def test_messages_for_scope_include_compact_model_context(self):
        block = Stage4FrontierBlock(
            id="indicator:pss_score",
            kind="indicator_decision",
            label="Choose likelihood for pss_score",
            construct_names=("stress",),
            variable_names=("pss_score",),
            payload={
                "variable": "pss_score",
                "fixed_distribution": "gaussian",
                "valid_links": ["identity"],
            },
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={
                "model_clock": "1d",
                "model_interval_days": 1.0,
                "outcome": "sleep",
                "latent_edges": [
                    {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Stress reduces subsequent sleep quality.",
                    }
                ],
            },
            distribution_cards=[
                {
                    "variable": "pss_score",
                    "construct": "stress",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "how_to_measure": "Use the pss column directly",
                    "options": [
                        {"distribution": "gaussian", "links": ["identity"]},
                    ],
                    "profile": {
                        "n_obs": 40,
                        "mean": 12.0,
                        "std": 3.5,
                        "min": 3.0,
                        "max": 21.0,
                    },
                    "validation_issues": [],
                }
            ],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "pss_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "how_to_measure": "Use the pss column directly",
                            "is_reference": True,
                            "has_distribution_decision_card": True,
                            "profile": {
                                "n_obs": 40,
                                "mean": 12.0,
                                "std": 3.5,
                                "min": 3.0,
                                "max": 21.0,
                            },
                        }
                    ],
                }
            ],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                }
            ],
        )
        plan = _make_plan(model_blocks=(block,))
        runtime = _make_runtime(plan)
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "## Model Topology" in user_content
        assert "Stress reduces subsequent sleep quality." not in user_content
        assert "Use the pss column directly" in user_content
        assert "model_interval_days" in user_content
        assert "### Construct Scale Cards" in user_content
        assert "see distribution decision card" in user_content
        assert "### Parameter Prior Cards" not in user_content
        assert "### Loading Constraints" not in user_content

    def test_messages_for_scope_render_frontier_contract(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={},
            distribution_cards=[
                {
                    "variable": "pss_score",
                    "construct": "stress",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "how_to_measure": "Use the pss column directly",
                    "options": [{"distribution": "gaussian", "links": ["identity"]}],
                    "profile": {"n_obs": 40},
                    "validation_issues": [],
                }
            ],
            loading_params=[
                {
                    "name": "lambda_pss_score_stress",
                    "construct": "stress",
                    "indicator": "pss_score",
                }
            ],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                }
            ],
            enable_literature=True,
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={"parameters": [{"name": "beta_stress_sleep"}]},
                authored_priors={
                    "beta_stress_sleep": {
                        "parameter": "beta_stress_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "accepted prior",
                    }
                },
            ),
        )
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "## Fixed Model Context" in user_content
        assert "## Frontier Status" in user_content
        assert "## Effect-Block Stability Discipline" in user_content
        assert "`id`: `effects:sleep`" in user_content
        assert "Use `submit_prior_block` with exactly this argument object:" in user_content
        assert (
            "This block owns one target construct's full incoming lagged-effect row."
            in user_content
        )
        assert "Treat the remaining headroom as advisory stability telemetry" in user_content
        assert "Normal(0, 0.1-0.2)" in user_content
        assert "Feedback Loop" in user_content
        assert "### Distribution Decision Cards" not in user_content
        assert "### Loading Constraints" not in user_content
        system_content = messages[0]["content"]
        assert "## Effect Row Budget Discipline" in system_content
        assert "batch those `search_literature` calls in the same turn" in system_content
        assert "advisory stability guidance" in system_content
        assert "stop searching and call `submit_prior_block`" in system_content

    def test_messages_for_scope_omit_literature_prompt_parts_when_disabled(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                }
            ],
            enable_literature=False,
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={"parameters": [{"name": "beta_stress_sleep"}]},
                authored_priors={},
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )

        system_content = messages[0]["content"]
        user_content = messages[1]["content"]

        assert "## Literature Evidence" not in system_content
        assert "search_literature" not in system_content
        assert "If you include non-empty `sources`" not in user_content

    def test_messages_for_scope_include_parameter_prior_cards(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress",),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[
                {
                    "variable": "worry_score",
                    "construct": "stress",
                    "measurement_dtype": "binary",
                    "aggregation": "mean",
                    "effective_window": "1d",
                    "how_to_measure": "Daily worry indicator",
                    "options": [{"distribution": "bernoulli", "links": ["logit", "probit"]}],
                    "profile": {"n_obs": 40},
                    "validation_issues": [],
                }
            ],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [],
                }
            ],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "expected_lag_days": 1.0,
                        "feedback_loop": True,
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={"parameters": [{"name": "beta_stress_sleep"}]},
                authored_priors={
                    "beta_stress_sleep": {
                        "parameter": "beta_stress_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "accepted prior",
                    }
                },
            ),
        )
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "### Parameter Prior Cards" in user_content
        assert "#### Current Accepted Priors" in user_content
        assert "Normal(mu=0.0, sigma=0.2)" in user_content
        assert "#### Fixed Effects" in user_content
        assert "| beta_stress_sleep | stress | sleep | lagged | 1.0 | yes | none |" in user_content
        assert "### Construct Scale Cards" in user_content
        assert "## Scope Snapshot" in user_content
        assert "Use `submit_prior_block` with exactly this argument object:" in user_content

    def test_messages_for_scope_include_accepted_coupled_priors_outside_local_scope(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[
                {
                    "variable": "worry_score",
                    "construct": "stress",
                    "measurement_dtype": "binary",
                    "aggregation": "mean",
                    "effective_window": "1d",
                    "how_to_measure": "Daily worry indicator",
                    "options": [{"distribution": "bernoulli", "links": ["logit", "probit"]}],
                    "profile": {"n_obs": 40},
                    "validation_issues": [],
                }
            ],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                },
                {
                    "parameter": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "structural_context": {"construct": "stress"},
                },
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={
                    "parameters": [
                        {"name": "beta_stress_sleep"},
                        {"name": "rho_stress"},
                    ]
                },
                authored_priors={
                    "beta_stress_sleep": {
                        "parameter": "beta_stress_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.2},
                        "sources": [],
                        "reasoning": "local accepted prior",
                    },
                    "rho_stress": {
                        "parameter": "rho_stress",
                        "distribution": "Beta",
                        "params": {"alpha": 3.0, "beta": 2.0},
                        "sources": [],
                        "reasoning": "coupled accepted prior",
                    },
                },
            ),
            last_validation_packet=make_stage4_validation_packet(
                status="prior_predictive_failure",
                feedback="PRIOR PREDICTIVE FEEDBACK:\nValidation FAILED",
                validation=AssemblyValidation(
                    compile_ok=True,
                    pp_checked=True,
                    pp_valid=False,
                    diagnostics=[
                        PriorValidationResult(
                            parameter="beta_stress_sleep",
                            is_valid=False,
                            code="scale_mismatch",
                            origin="prior_predictive",
                            issue="Effect prior mismatches the stress dynamics scale.",
                            suggested_adjustment="Reconcile the effect prior with stress persistence.",
                            related_parameters=["rho_stress"],
                        )
                    ],
                ),
                active_scope_id=block.id,
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "Accepted Coupled Priors Outside This Edit Scope" in user_content
        assert "Beta(alpha=3.0, beta=2.0)" in user_content
        assert "`rho_stress`" in user_content

    def test_messages_for_effect_scope_include_structural_coupled_dynamics_priors_pre_failure(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
            payload={"target_construct": "sleep", "cause_names": ("stress",)},
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[
                {
                    "variable": "worry_score",
                    "construct": "stress",
                    "measurement_dtype": "binary",
                    "aggregation": "mean",
                    "effective_window": "1d",
                    "how_to_measure": "Daily worry indicator",
                    "options": [{"distribution": "bernoulli", "links": ["logit", "probit"]}],
                    "profile": {"n_obs": 40},
                    "validation_issues": [],
                }
            ],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                },
                {
                    "parameter": "rho_stress",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "structural_context": {"construct": "stress"},
                },
                {
                    "parameter": "sigma_sleep",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "structural_context": {"construct": "sleep"},
                },
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={
                    "parameters": [
                        {"name": "beta_stress_sleep"},
                        {"name": "rho_stress"},
                        {"name": "sigma_sleep"},
                    ]
                },
                authored_priors={
                    "rho_stress": {
                        "parameter": "rho_stress",
                        "distribution": "Beta",
                        "params": {"alpha": 3.0, "beta": 2.0},
                        "sources": [],
                        "reasoning": "stress persistence prior",
                    },
                    "sigma_sleep": {
                        "parameter": "sigma_sleep",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.3},
                        "sources": [],
                        "reasoning": "sleep innovation scale prior",
                    },
                },
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "Accepted Coupled Priors Outside This Edit Scope" in user_content
        assert "`rho_stress`" in user_content
        assert "`sigma_sleep`" in user_content
        assert "Beta(alpha=3.0, beta=2.0)" in user_content
        assert "HalfNormal(sigma=0.3)" in user_content

    def test_messages_for_dynamics_scope_include_budget_discipline(self):
        block = Stage4FrontierBlock(
            id="dynamics:sleep",
            kind="dynamics_prior",
            label="Dynamics prior",
            construct_names=("sleep",),
            parameter_names=("rho_sleep", "sigma_sleep"),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "sleep",
                    "description": "Sleep quality",
                    "role": "endogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": True,
                    "reference_indicator": "sleep_quality",
                    "indicators": [],
                }
            ],
            prior_cards=[
                {
                    "parameter": "rho_sleep",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "structural_context": {"construct": "sleep"},
                },
                {
                    "parameter": "sigma_sleep",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "structural_context": {"construct": "sleep"},
                },
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={
                    "parameters": [
                        {"name": "rho_sleep"},
                        {"name": "sigma_sleep"},
                    ]
                }
            ),
        )
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]
        system_content = messages[0]["content"]

        assert "clear damping headroom for later incoming effects" in user_content
        assert "## Dynamics Budget Discipline" in system_content
        assert "baseline persistence absent incoming feedback" in system_content

    def test_messages_for_dynamics_scope_include_structural_coupled_effect_priors_pre_failure(self):
        block = Stage4FrontierBlock(
            id="dynamics:sleep",
            kind="dynamics_prior",
            label="Dynamics prior",
            construct_names=("sleep",),
            parameter_names=("rho_sleep", "sigma_sleep"),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "rho_sleep",
                    "role": "ar_coefficient",
                    "constraint": "unit_interval",
                    "structural_context": {"construct": "sleep"},
                },
                {
                    "parameter": "sigma_sleep",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "structural_context": {"construct": "sleep"},
                },
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                    },
                },
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={
                    "parameters": [
                        {"name": "rho_sleep"},
                        {"name": "sigma_sleep"},
                        {"name": "beta_stress_sleep"},
                    ]
                },
                authored_priors={
                    "beta_stress_sleep": {
                        "parameter": "beta_stress_sleep",
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 0.15},
                        "sources": [],
                        "reasoning": "incoming effect prior",
                    }
                },
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "Accepted Coupled Priors Outside This Edit Scope" in user_content
        assert "`beta_stress_sleep`" in user_content
        assert "Normal(mu=0.0, sigma=0.15)" in user_content

    def test_messages_for_effect_scope_include_neighboring_topology_context(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("stress", "sleep"),
            parameter_names=("beta_stress_sleep",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={
                "model_clock": "1d",
                "model_interval_days": 1.0,
                "outcome": "sleep",
                "latent_edges": [
                    {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Primary effect under review",
                    },
                    {
                        "cause": "mood",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Competing parent of sleep",
                    },
                ],
            },
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_stress_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "expected_lag_days": 1.0,
                        "feedback_loop": False,
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={"parameters": [{"name": "beta_stress_sleep"}]}
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "| stress | sleep | yes | Primary effect under review |" in user_content
        assert "| mood | sleep | yes | Competing parent of sleep |" in user_content

    def test_messages_for_effect_scope_include_feedback_scc_membership_summary(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            construct_names=("activity", "sleep"),
            parameter_names=("beta_activity_sleep",),
            payload={"target_construct": "sleep", "cause_names": ("activity",)},
        )
        msgs = Stage4Messages(
            question="Does activity affect sleep?",
            model_topology={
                "model_clock": "1d",
                "model_interval_days": 1.0,
                "outcome": "sleep",
                "latent_edges": [
                    {
                        "cause": "activity",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Incoming edge under review",
                    },
                    {
                        "cause": "sleep",
                        "effect": "mood",
                        "lagged": True,
                        "description": "Cycle continuation",
                    },
                    {
                        "cause": "mood",
                        "effect": "stress",
                        "lagged": True,
                        "description": "Cycle continuation",
                    },
                    {
                        "cause": "stress",
                        "effect": "activity",
                        "lagged": True,
                        "description": "Cycle closure",
                    },
                ],
            },
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_activity_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "activity",
                        "effect": "sleep",
                        "lagged": True,
                        "expected_lag_days": 1.0,
                        "feedback_loop": True,
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={"parameters": [{"name": "beta_activity_sleep"}]}
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "### SCC Membership" in user_content
        assert "| sleep, activity | activity, sleep, mood, stress | yes |" in user_content

    def test_messages_for_narrowed_effect_repair_scope_do_not_reexpand_neighboring_topology(self):
        block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior (internal SCC parameters only)",
            construct_names=("activity", "sleep"),
            parameter_names=("beta_activity_sleep",),
            expand_neighbor_topology=False,
        )
        msgs = Stage4Messages(
            question="Does activity affect sleep?",
            model_topology={
                "model_clock": "1d",
                "model_interval_days": 1.0,
                "outcome": "sleep",
                "latent_edges": [
                    {
                        "cause": "stress",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Outside-SCC parent that should stay hidden",
                    },
                    {
                        "cause": "activity",
                        "effect": "sleep",
                        "lagged": True,
                        "description": "Internal SCC edge under repair",
                    },
                    {
                        "cause": "sleep",
                        "effect": "activity",
                        "lagged": True,
                        "description": "Reciprocal SCC edge under repair",
                    },
                ],
            },
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "beta_activity_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "activity",
                        "effect": "sleep",
                        "lagged": True,
                        "expected_lag_days": 1.0,
                        "feedback_loop": True,
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={"parameters": [{"name": "beta_activity_sleep"}]}
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "| activity | sleep | yes | Internal SCC edge under repair |" in user_content
        assert "| sleep | activity | yes | Reciprocal SCC edge under repair |" in user_content
        assert "Outside-SCC parent that should stay hidden" not in user_content
        assert "| stress | sleep | yes |" not in user_content

    def test_messages_for_correlation_scope_include_scale_coupling_context_and_guidance(self):
        block = Stage4FrontierBlock(
            id="correlation:cor_sleep_stress",
            kind="correlation_prior",
            label="Correlation prior",
            construct_names=("sleep", "stress"),
            parameter_names=("cor_sleep_stress",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[],
            loading_params=[],
            construct_scale_cards=[],
            prior_cards=[
                {
                    "parameter": "cor_sleep_stress",
                    "role": "correlation",
                    "constraint": "correlation",
                    "structural_context": {
                        "construct_1": "sleep",
                        "construct_2": "stress",
                        "dependency_kind": "diffusion",
                    },
                },
                {
                    "parameter": "sigma_sleep",
                    "role": "residual_sd",
                    "constraint": "positive",
                    "structural_context": {"construct": "sleep"},
                },
                {
                    "parameter": "t0_sd_stress",
                    "role": "initial_state_sd",
                    "constraint": "positive",
                    "structural_context": {"construct": "stress"},
                },
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={
                    "parameters": [
                        {"name": "cor_sleep_stress"},
                        {"name": "sigma_sleep"},
                        {"name": "t0_sd_stress"},
                    ]
                },
                authored_priors={
                    "sigma_sleep": {
                        "parameter": "sigma_sleep",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.25},
                        "sources": [],
                        "reasoning": "sleep scale prior",
                    },
                    "t0_sd_stress": {
                        "parameter": "t0_sd_stress",
                        "distribution": "HalfNormal",
                        "params": {"sigma": 0.8},
                        "sources": [],
                        "reasoning": "stress initial-state scale prior",
                    },
                },
            ),
        )

        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        system_content = messages[0]["content"]
        user_content = messages[1]["content"]

        assert "## Continuous-Time Dynamics" in system_content
        assert "## Initial-State Scale Discipline" in system_content
        assert "Accepted Coupled Priors Outside This Edit Scope" in user_content
        assert "`sigma_sleep`" in user_content
        assert "`t0_sd_stress`" in user_content

    def test_messages_for_scope_respects_visible_sections_even_when_data_matches(self):
        block = Stage4FrontierBlock(
            id="review:model_spec",
            kind="global_review",
            label="Model review",
            construct_names=("stress",),
            parameter_names=("lambda_worry_stress",),
            payload={"reopenable_block_ids": ("indicator:worry_score",)},
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={},
            distribution_cards=[
                {
                    "variable": "worry_score",
                    "construct": "stress",
                    "measurement_dtype": "continuous",
                    "aggregation": "mean",
                    "how_to_measure": "Use the worry column directly",
                    "options": [{"distribution": "gaussian", "links": ["identity"]}],
                    "profile": {"n_obs": 40},
                    "validation_issues": [],
                }
            ],
            loading_params=[
                {
                    "name": "lambda_worry_stress",
                    "construct": "stress",
                    "indicator": "worry_score",
                    "constraint": "positive",
                }
            ],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [],
                }
            ],
            prior_cards=[
                {
                    "parameter": "lambda_worry_stress",
                    "role": "loading",
                    "constraint": "positive",
                    "structural_context": {
                        "construct": "stress",
                        "indicator": "worry_score",
                        "reference_indicator": "pss_score",
                    },
                }
            ],
        )
        plan = _make_plan(review_block=block)
        runtime = _make_runtime(plan)
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "### Loading Orientation" in user_content
        assert "### Construct Scale Cards" in user_content
        assert "### Distribution Decision Cards" not in user_content
        assert "### Parameter Prior Cards" not in user_content

    def test_messages_for_scope_render_extended_profile_and_support_window_metadata(self):
        block = Stage4FrontierBlock(
            id="indicator:worry_score",
            kind="indicator_decision",
            label="Indicator decision",
            construct_names=("stress",),
            variable_names=("worry_score",),
            payload={
                "variable": "worry_score",
                "fixed_distribution": "bernoulli",
                "valid_links": ["logit", "probit"],
            },
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={},
            distribution_cards=[
                {
                    "variable": "worry_score",
                    "construct": "stress",
                    "measurement_dtype": "binary",
                    "aggregation": "mean",
                    "effective_window": "1w",
                    "how_to_measure": "Weekly worry indicator",
                    "options": [{"distribution": "bernoulli", "links": ["logit", "probit"]}],
                    "profile": {
                        "n_obs": 40,
                        "q50": 1.0,
                        "time_coverage_ratio": 0.75,
                        "max_gap_ratio": 2.5,
                        "duplicate_pct": 0.05,
                        "n_unparseable_timestamps": 1,
                    },
                    "validation_issues": [],
                }
            ],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "worry_score",
                            "measurement_dtype": "binary",
                            "aggregation": "mean",
                            "effective_window": "1w",
                            "is_reference": False,
                            "has_distribution_decision_card": True,
                            "profile": {
                                "n_obs": 40,
                                "q50": 1.0,
                                "time_coverage_ratio": 0.75,
                            },
                            "how_to_measure": "Weekly worry indicator",
                        }
                    ],
                }
            ],
            prior_cards=[],
        )
        plan = _make_plan(model_blocks=(block,))
        runtime = _make_runtime(plan)
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "| worry_score | stress | binary | mean | 1w |" in user_content
        assert "q50=1" in user_content
        assert "coverage=75%" in user_content
        assert "max_gap=2.5x" in user_content
        assert "dups=5.0%" in user_content
        assert "bad_ts=1" in user_content

    def test_messages_for_scope_render_stateful_accepted_decisions(self):
        block = Stage4FrontierBlock(
            id="measurement:stress",
            kind="measurement_prior",
            label="Measurement prior",
            construct_names=("stress",),
            variable_names=("pss_score", "worry_score"),
            parameter_names=("lambda_worry_score_stress",),
        )
        msgs = Stage4Messages(
            question="Does stress affect sleep?",
            model_topology={"model_clock": "1d", "model_interval_days": 1.0, "outcome": "sleep"},
            distribution_cards=[
                {
                    "variable": "worry_score",
                    "construct": "stress",
                    "measurement_dtype": "binary",
                    "aggregation": "mean",
                    "effective_window": "1d",
                    "how_to_measure": "Daily worry indicator",
                    "options": [{"distribution": "bernoulli", "links": ["logit", "probit"]}],
                    "profile": {"n_obs": 40},
                    "validation_issues": [],
                }
            ],
            loading_params=[],
            construct_scale_cards=[
                {
                    "construct": "stress",
                    "description": "Perceived stress",
                    "role": "exogenous",
                    "temporal_status": "time_varying",
                    "is_outcome": False,
                    "reference_indicator": "pss_score",
                    "indicators": [
                        {
                            "indicator": "pss_score",
                            "measurement_dtype": "continuous",
                            "aggregation": "mean",
                            "effective_window": "1d",
                            "is_reference": True,
                            "has_distribution_decision_card": False,
                            "profile": {"n_obs": 40},
                            "how_to_measure": "Daily PSS score",
                        },
                        {
                            "indicator": "worry_score",
                            "measurement_dtype": "binary",
                            "aggregation": "mean",
                            "effective_window": "1d",
                            "is_reference": False,
                            "has_distribution_decision_card": True,
                            "profile": {"n_obs": 40},
                            "how_to_measure": "Daily worry indicator",
                        },
                    ],
                }
            ],
            prior_cards=[
                {
                    "parameter": "lambda_worry_score_stress",
                    "role": "loading",
                    "constraint": "positive",
                    "structural_context": {
                        "construct": "stress",
                        "indicator": "worry_score",
                        "reference_indicator": "pss_score",
                    },
                }
            ],
        )
        plan = _make_plan(prior_blocks=(block,))
        runtime = _make_runtime(
            plan,
            phase="prior_blocks",
            active_block_id=block.id,
            accepted=Stage4AcceptedArtifacts(
                model_spec={
                    "likelihoods": [
                        {
                            "variable": "pss_score",
                            "distribution": "gaussian",
                            "link": "identity",
                        },
                        {
                            "variable": "worry_score",
                            "distribution": "bernoulli",
                            "link": "probit",
                        },
                    ],
                    "parameters": [
                        {
                            "name": "lambda_worry_score_stress",
                            "role": "loading",
                            "constraint": "positive",
                        }
                    ],
                }
            ),
        )
        messages = msgs.messages_for_block(
            block,
            plan,
            runtime,
            get_stage4_block_handler(block.kind),
        )
        user_content = messages[1]["content"]

        assert "### Distribution Decision Cards" in user_content
        assert "`bernoulli` / `probit`" in user_content
        assert (
            "| lambda_worry_score_stress | stress | worry_score | pss_score | positive |"
            in user_content
        )

    def test_stage4_turn_exposes_tools_by_block_kind(self):
        model_block = Stage4FrontierBlock(
            id="indicator:steps",
            kind="indicator_decision",
            label="Indicator decision",
            variable_names=("steps",),
            payload={
                "variable": "steps",
                "dtype": "count",
                "valid_distributions": ["negative_binomial", "poisson"],
                "link_options": {
                    "negative_binomial": ["log"],
                    "poisson": ["log"],
                },
            },
        )
        measurement_block = Stage4FrontierBlock(
            id="measurement:activity",
            kind="measurement_prior",
            label="Measurement prior",
            parameter_names=("lambda_steps_activity",),
            construct_names=("activity",),
            variable_names=("steps",),
        )
        prior_block = Stage4FrontierBlock(
            id="effects:sleep",
            kind="effect_prior",
            label="Effect prior",
            parameter_names=("beta_activity_sleep",),
        )
        plan = _make_plan(
            model_blocks=(model_block,),
            prior_blocks=(measurement_block, prior_block),
        )
        runtime = _make_runtime(plan)
        session = _make_stage4_session(
            question="Does activity improve sleep?",
            plan=plan,
            runtime=runtime,
            skeleton=Stage4Skeleton(),
            causal_spec={},
            stage4_grounding_fn=lambda *_args, **_kwargs: pytest.fail("grounding should not run"),
            prior_cards=[
                {
                    "parameter": "lambda_steps_activity",
                    "role": "loading",
                    "constraint": "positive",
                    "structural_context": {
                        "construct": "activity",
                        "indicator": "steps",
                        "reference_indicator": "steps_ref",
                    },
                },
                {
                    "parameter": "beta_activity_sleep",
                    "role": "fixed_effect",
                    "constraint": "none",
                    "structural_context": {
                        "cause": "activity",
                        "effect": "sleep",
                        "lagged": True,
                    },
                },
            ],
            enable_literature=True,
            enable_paraphrasing=True,
        )
        tools = [
            SimpleNamespace(name="submit_indicator_choice"),
            SimpleNamespace(name="submit_prior_block"),
            SimpleNamespace(name="search_literature"),
            SimpleNamespace(name="elicit_prior_gmm"),
        ]
        turn = session.current_turn()
        assert turn is not None
        assert [tool.name for tool in tools if tool.name in turn.allowed_tool_names] == [
            "submit_indicator_choice"
        ]

        runtime.domain.accepted.model_spec = {
            "parameters": [
                {"name": "lambda_steps_activity", "role": "loading"},
                {"name": "beta_activity_sleep"},
            ]
        }
        _set_runtime_block(plan, runtime, "measurement:activity")

        turn = session.current_turn()
        assert turn is not None
        assert [tool.name for tool in tools if tool.name in turn.allowed_tool_names] == [
            "submit_prior_block",
            "elicit_prior_gmm",
        ]

        _set_runtime_block(plan, runtime, "effects:sleep")

        turn = session.current_turn()
        assert turn is not None
        assert [tool.name for tool in tools if tool.name in turn.allowed_tool_names] == [
            "submit_prior_block",
            "search_literature",
            "elicit_prior_gmm",
        ]


