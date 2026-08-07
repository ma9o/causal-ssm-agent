import marimo

__generated_with = "0.23.11"
app = marimo.App(width="medium")


@app.cell
def imports_marimo():
    import marimo as mo

    return (mo,)


@app.cell
def imports():
    import json
    from pathlib import Path

    import case_study_support as cs
    import polars as pl
    import prior_specification_support as ps

    from nof1_causal_lab.flows.transitions.measurement_structure.assemble import (
        build_causal_design,
    )
    from nof1_causal_lab.models.ssm.construct_admission import build_construct_units
    from nof1_causal_lab.models.structural import build_structural_plan
    from nof1_causal_lab.utils.causal_design import build_reference_indicator_lookup
    from nof1_causal_lab.utils.structural_plan import (
        get_known_inputs,
        get_manifest_indicators,
        get_plan_constructs,
        get_plan_indicators,
    )

    return (
        Path,
        build_causal_design,
        build_construct_units,
        build_reference_indicator_lookup,
        build_structural_plan,
        cs,
        get_known_inputs,
        get_manifest_indicators,
        get_plan_constructs,
        get_plan_indicators,
        json,
        pl,
        ps,
    )


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Codex-driven production prior specification

    This is an **authoring workbench**, not another embedded agent. The `PROPOSALS` cell below
    is the judgment surface: Codex edits ordinary Python payloads, reads the resulting
    diagnostics, and revises them directly.

    Its primary purpose is to validate the authoring framework implemented in `src` by exercising
    it manually, one construct at a time, in isolation from the surrounding agent and workflow
    infrastructure. At each step, the human-guided authoring process is the reference against which
    we evaluate whether the production prompt exposes the right information and parameter surfaces,
    the compiler interprets the proposal correctly, and the admission diagnostics lead toward a
    scientifically defensible revision. A rejected proposal is not automatically an authoring
    mistake: it may instead reveal a framework bug, an unnecessarily restrictive authoring surface,
    or misleading guidance.

    Everything after input loading is production code. Each payload goes through
    `ConstructBuildState.submit_construct`, so parameter membership, likelihood compatibility,
    compiler binding, exact nonlinear Diffrax prior prediction, and the C1--C5 reachability
    battery are the same ones used by Stage 4. Once all constructs are admitted, the notebook
    runs the same shared-draw full-model barrier. It uses the production immutable-checkpoint
    reducer and content-addressed admission-evaluation cache, but does **not** call Temporal,
    `ArtifactStore`, telemetry, or the Pi harness.

    The current DEMO artifact files are read directly only to snapshot the four Stage-4 inputs
    into memory while storage is being refactored. The stored causal-design snapshot predates both
    known-input authoring and the structural compiler. This trial re-derives it in memory, compiles
    observed time-invariant constructs with direct dynamic effects (including CYP2C19 capacity) as
    known inputs, marks the remaining measured baseline context scientific-only, and then runs the
    production `CausalDesign -> StructuralPlan` compiler. That explicit reduction removes
    unresolved static-target chains instead of relying on the old silent edge drop. The stored
    artifacts are not mutated.
    """)
    return


@app.cell
def input_paths(Path):
    WORKSPACE_STORE = Path(__file__).resolve().parents[3] / "data/DEMO/store"
    WORKSPACE_ID = WORKSPACE_STORE.parent.name
    QUESTION_PATH = WORKSPACE_STORE / "question/v1/question.json"
    CAUSAL_DESIGN_PATH = WORKSPACE_STORE / "causal_design/v1/causal_design.json"
    PANEL_PATH = WORKSPACE_STORE / "panel/v1/panel.parquet"
    VALIDATION_REPORT_PATH = WORKSPACE_STORE / "validation_report/v1/validation_report.json"
    return (
        CAUSAL_DESIGN_PATH,
        PANEL_PATH,
        QUESTION_PATH,
        VALIDATION_REPORT_PATH,
        WORKSPACE_ID,
        WORKSPACE_STORE,
    )


@app.cell
def load_input_snapshot(
    CAUSAL_DESIGN_PATH,
    PANEL_PATH,
    QUESTION_PATH,
    VALIDATION_REPORT_PATH,
    build_causal_design,
    build_reference_indicator_lookup,
    build_structural_plan,
    json,
    pl,
):
    question = json.loads(QUESTION_PATH.read_text())["text"]
    _stored_causal_design = json.loads(CAUSAL_DESIGN_PATH.read_text())["causal_design"]
    _latent = _stored_causal_design["latent"]
    _measurement = _stored_causal_design["measurement"]
    _reference_indicators = build_reference_indicator_lookup(_measurement["indicators"])
    _observed_names = {indicator["construct_name"] for indicator in _measurement["indicators"]}
    _dynamic_observed = {
        construct["name"]
        for construct in _latent["constructs"]
        if construct.get("temporal_status") == "time_varying"
        and construct["name"] in _observed_names
    }
    _static_observed = {
        construct["name"]
        for construct in _latent["constructs"]
        if construct.get("temporal_status") == "time_invariant"
        and construct["name"] in _reference_indicators
    }
    _static_input_names = {
        edge["cause"]
        for edge in _latent["edges"]
        if edge["cause"] in _static_observed and edge["effect"] in _dynamic_observed
    }
    _known_inputs = [
        {
            "construct": construct_name,
            "source_indicator": _reference_indicators[construct_name],
            "scale": 1.0,
            "missing_policy": "forward_fill",
        }
        for construct_name in sorted(_static_input_names)
    ]
    _known_inputs.append(
        {
            "construct": "counterfactual_taper_regime",
            "source_indicator": "observed_taper_regime_indicator",
            "scale": 1.0,
            "missing_policy": "forward_fill",
        }
    )
    _scientific_only_constructs = [
        {
            "construct": construct_name,
            "reason": (
                "Measured stable context has no direct effect on a retained dynamic state; "
                "its static chain is not an estimable N-of-1 structural equation."
            ),
        }
        for construct_name in sorted(_static_observed - _static_input_names)
    ]
    _causal_design = build_causal_design(
        _latent,
        _measurement,
        _stored_causal_design.get("identifiability"),
        known_inputs=_known_inputs,
        scientific_only_constructs=_scientific_only_constructs,
    )
    structural_plan = build_structural_plan(_causal_design)
    data_for_model = pl.read_parquet(PANEL_PATH)
    validation_report = json.loads(VALIDATION_REPORT_PATH.read_text())
    indicator_audits = validation_report["indicators"]
    return structural_plan, data_for_model, indicator_audits, question, validation_report


@app.cell(hide_code=True)
def input_audit(
    build_construct_units,
    structural_plan,
    data_for_model,
    get_known_inputs,
    get_plan_constructs,
    get_plan_indicators,
    mo,
    validation_report,
):
    _units = build_construct_units(structural_plan)
    _construct_count = len(get_plan_constructs(structural_plan))
    _indicator_count = len(get_plan_indicators(structural_plan))
    _feedback_units = [unit for unit in _units if len(unit.constructs) > 1]
    _errors = [
        issue
        for audit in validation_report["indicators"].values()
        for issue in audit["validation"]["issues"]
        if issue["severity"] == "error"
    ]
    _feedback_summary = (
        ", ".join(f"{len(unit.constructs)} members" for unit in _feedback_units) or "none"
    )
    mo.md(
        f"""
        ## Input snapshot

        | surface | value |
        |---|---:|
        | panel rows | {data_for_model.height:,} |
        | constructs | {_construct_count} |
        | indicators | {_indicator_count} |
        | admission units | {len(_units)} |
        | feedback components | {_feedback_summary} |
        | known transition inputs | {len(get_known_inputs(structural_plan))} |
        | validation status | `{validation_report["is_valid"]}` |
        | indicator-level validation errors | {len(_errors)} |

        The invalid validation report is intentionally left visible: a workbench meant to catch
        framework-wide omissions must not silently sanitize an upstream state that production can
        currently hand to Stage 4.
        """
    )
    return


@app.cell(hide_code=True)
def authoring_protocol(mo):
    mo.md(r"""
    ## Authoring protocol

    - Keep `N_DRAWS = 16` while discovering compiler or framework failures; use `200` for the
      production-strength replay and full-model barrier.
    - Append exactly one proposal at a time. If it is rejected, append a revised payload for the
      same construct; replay starts from a fresh state on every edit.
    - Before revising a rejected proposal, classify the failure: authored judgment error, framework
      bug, framework limitation, or misleading prompt/diagnostic.
    - Do not distort a scientifically defensible likelihood or prior merely to clear an admission
      check. If the framework is driving the author incorrectly, fix the production implementation
      in `src`, record the finding in the issue ledger, and replay the unchanged proposal first.
    - Accept a soft check only with its exact `(check, target)` pair and a substantive rationale.
      Hard checks are never overridden.
    - Do not author compiler defaults or parameters absent from the production prompt shown below.
    - Parameters marked conditional in the production prompt are authorable surfaces, not mandatory
      priors: include them only when the submitted family and link activate them.
    - Time-invariant observed quantities are conditioned on through the compiled known-input
      surface in this N-of-1 trial; they are not authored as latent state priors.
    """)
    return


@app.cell(hide_code=True)
def prior_replay_archive():
    _N_DRAWS = 16
    _SEED = 20260715

    _PROPOSALS = [
        {
            "construct": "external_stressful_events",
            "indicators": [
                {
                    "variable": "external_stressor_event_count",
                    "family": "poisson",
                    "link": "log",
                    "reasoning": (
                        "A daily count with low mean and only modest overdispersion; the first "
                        "workbench pass keeps the observation model parsimonious."
                    ),
                }
            ],
            "priors": {
                "manifest_mean_external_stressor_event_count": {
                    "distribution": "Normal",
                    "params": {"mu": -2.94, "sigma": 0.8},
                    "reasoning": (
                        "Centers the baseline Poisson rate near the observed 0.053 events/day "
                        "while retaining substantial uncertainty for sparse event evidence."
                    ),
                },
                "rho_external_stressful_events": {
                    "distribution": "Beta",
                    "params": {"alpha": 2.0, "beta": 5.0},
                    "reasoning": (
                        "External shocks are primarily day-specific, with limited persistence "
                        "beyond the event day."
                    ),
                },
                "sigma_external_stressful_events": {
                    "distribution": "LogNormal",
                    "params": {"mu": -0.6, "sigma": 0.35},
                    "reasoning": (
                        "Keeps the latent innovation on an approximately standardized scale "
                        "without allowing sparse shocks to dominate the count link."
                    ),
                },
            },
        },
        {
            "construct": "external_stressful_events",
            "indicators": [
                {
                    "variable": "external_stressor_event_count",
                    "family": "poisson",
                    "link": "log",
                    "reasoning": (
                        "A daily count with low mean and only modest overdispersion; the first "
                        "workbench pass keeps the observation model parsimonious."
                    ),
                }
            ],
            "priors": {
                "manifest_mean_external_stressor_event_count": {
                    "distribution": "Normal",
                    "params": {"mu": -3.35, "sigma": 0.5},
                    "reasoning": (
                        "Offsets log-normal mean inflation after widening the latent to its "
                        "standardized scale while keeping the marginal rate near 0.053/day."
                    ),
                },
                "rho_external_stressful_events": {
                    "distribution": "Beta",
                    "params": {"alpha": 2.0, "beta": 5.0},
                    "reasoning": (
                        "External shocks are primarily day-specific, with limited persistence "
                        "beyond the event day."
                    ),
                },
                "sigma_external_stressful_events": {
                    "distribution": "LogNormal",
                    "params": {"mu": 0.35, "sigma": 0.3},
                    "reasoning": (
                        "Raises the stationary latent spread to approximately one so the state "
                        "respects the standardized-latent convention."
                    ),
                },
            },
        },
        {
            "construct": "counterfactual_taper_regime",
            "indicators": [
                {
                    "variable": "observed_taper_regime_indicator",
                    "family": "bernoulli",
                    "link": "logit",
                    "reasoning": "Binary documented taper versus maintenance regime.",
                }
            ],
            "priors": {
                "manifest_mean_observed_taper_regime_indicator": {
                    "distribution": "Normal",
                    "params": {"mu": -1.6, "sigma": 0.6},
                    "reasoning": (
                        "Centers the marginal taper prevalence near the observed 22% after "
                        "mixing over the latent regime state."
                    ),
                },
                "rho_counterfactual_taper_regime": {
                    "distribution": "Beta",
                    "params": {"alpha": 48.0, "beta": 2.0},
                    "reasoning": (
                        "A treatment regime persists for weeks rather than changing daily."
                    ),
                },
                "sigma_counterfactual_taper_regime": {
                    "distribution": "LogNormal",
                    "params": {"mu": -1.25, "sigma": 0.3},
                    "reasoning": (
                        "Calibrates a slow process to approximately unit stationary latent scale."
                    ),
                },
            },
        },
        {
            "construct": "counterfactual_taper_regime",
            "indicators": [
                {
                    "variable": "observed_taper_regime_indicator",
                    "family": "bernoulli",
                    "link": "logit",
                    "reasoning": "Binary documented taper versus maintenance regime.",
                }
            ],
            "priors": {
                "manifest_mean_observed_taper_regime_indicator": {
                    "distribution": "Normal",
                    "params": {"mu": -1.6, "sigma": 0.6},
                    "reasoning": (
                        "Centers the marginal taper prevalence near the observed 22% after "
                        "mixing over the latent regime state."
                    ),
                },
                "rho_counterfactual_taper_regime": {
                    "distribution": "Beta",
                    "params": {"alpha": 48.0, "beta": 2.0},
                    "reasoning": (
                        "A treatment regime persists for weeks rather than changing daily."
                    ),
                },
                "sigma_counterfactual_taper_regime": {
                    "distribution": "LogNormal",
                    "params": {"mu": -1.25, "sigma": 0.3},
                    "reasoning": (
                        "Calibrates a slow process to approximately unit stationary latent scale."
                    ),
                },
            },
        },
        {
            "construct": "genetic_family_liability",
            "indicators": [
                {
                    "variable": "family_psychiatric_liability_mentions",
                    "family": "poisson",
                    "link": "log",
                    "reasoning": "Count of distinct documented family-liability categories.",
                }
            ],
            "priors": {
                "manifest_mean_family_psychiatric_liability_mentions": {
                    "distribution": "Normal",
                    "params": {"mu": -0.72, "sigma": 0.7},
                    "reasoning": (
                        "Centers the marginal count near 0.8 after mixing over a unit-scale "
                        "static trait."
                    ),
                },
                "t0_mean_genetic_family_liability": {
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.5},
                    "reasoning": (
                        "Centers this subject-specific static trait on the standardized latent "
                        "origin."
                    ),
                },
                "t0_sd_genetic_family_liability": {
                    "distribution": "LogNormal",
                    "params": {"mu": 0.0, "sigma": 0.3},
                    "reasoning": (
                        "Unit-scale uncertainty for a weakly observed time-invariant trait."
                    ),
                },
            },
        },
        {
            "construct": "genetic_family_liability",
            "indicators": [
                {
                    "variable": "family_psychiatric_liability_mentions",
                    "family": "poisson",
                    "link": "log",
                    "reasoning": "Count of distinct documented family-liability categories.",
                }
            ],
            "priors": {
                "manifest_mean_family_psychiatric_liability_mentions": {
                    "distribution": "Normal",
                    "params": {"mu": -0.72, "sigma": 0.7},
                    "reasoning": (
                        "Centers the marginal count near 0.8 after mixing over a unit-scale "
                        "static trait."
                    ),
                },
                "t0_mean_genetic_family_liability": {
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.5},
                    "reasoning": (
                        "Centers this subject-specific static trait on the standardized latent "
                        "origin."
                    ),
                },
                "t0_sd_genetic_family_liability": {
                    "distribution": "LogNormal",
                    "params": {"mu": 0.0, "sigma": 0.3},
                    "reasoning": (
                        "Unit-scale uncertainty for a weakly observed time-invariant trait."
                    ),
                },
            },
        },
        {
            "construct": "early_life_adversity",
            "indicators": [
                {
                    "variable": "early_life_adversity_history",
                    "family": "ordered_logistic",
                    "link": "cumulative_logit",
                    "reasoning": "Four ordered adversity-history levels.",
                }
            ],
            "priors": {
                "manifest_mean_early_life_adversity_history": {
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 1.0},
                    "reasoning": "Weakly informative ordinal-location surface.",
                },
                "t0_mean_early_life_adversity": {
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.5},
                    "reasoning": "Centers the static state on the standardized latent origin.",
                },
                "t0_sd_early_life_adversity": {
                    "distribution": "LogNormal",
                    "params": {"mu": 0.0, "sigma": 0.3},
                    "reasoning": (
                        "Uses the pooled static-state scale family already established by "
                        "genetic family liability."
                    ),
                },
            },
        },
    ]
    return


@app.cell(hide_code=True)
def pre_recompile_proposal_archive():
    _N_DRAWS = 16
    _SEED = 20260716
    _PROPOSALS = [
        proposal
        for proposal in [
            {
                "construct": "external_stressful_events",
                "indicators": [
                    {
                        "variable": "external_stressor_event_count",
                        "family": "poisson",
                        "link": "log",
                        "reasoning": (
                            "The channel is a nonnegative integer count with 95.4% zeros and only "
                            "modest variance-to-mean inflation (1.39), so Poisson is the "
                            "parsimonious compatible emission for the first exact PPC pass."
                        ),
                    }
                ],
                "priors": {
                    "manifest_mean_external_stressor_event_count": {
                        "distribution": "Normal",
                        "params": {"mu": -3.35, "sigma": 0.5},
                        "reasoning": (
                            "The raw mean is 0.053/day; this intercept is below log(0.053) to "
                            "allow for marginal mean inflation from the unit-scale latent on the "
                            "log link."
                        ),
                    },
                    "rho_external_stressful_events": {
                        "distribution": "Beta",
                        "params": {"alpha": 2.0, "beta": 5.0},
                        "reasoning": (
                            "External shocks should mostly be day-specific, with limited carryover "
                            "relative to the one-day model clock."
                        ),
                    },
                    "sigma_external_stressful_events": {
                        "distribution": "LogNormal",
                        "params": {"mu": 0.35, "sigma": 0.3},
                        "reasoning": (
                            "Pairs a relatively fast relaxation prior with enough innovation to "
                            "reach the compiler's standardized latent scale."
                        ),
                    },
                },
            },
            {
                "construct": "genetic_family_liability",
                "indicators": [
                    {
                        "variable": "family_psychiatric_liability_mentions",
                        "family": "poisson",
                        "link": "log",
                        "reasoning": (
                            "The five observations are nonnegative integer category counts with "
                            "variance close to the mean; Poisson preserves the declared count "
                            "support while acknowledging that the tiny sample cannot identify "
                            "overdispersion reliably."
                        ),
                    }
                ],
                "priors": {
                    "manifest_mean_family_psychiatric_liability_mentions": {
                        "distribution": "Normal",
                        "params": {"mu": -0.72, "sigma": 0.7},
                        "reasoning": (
                            "Centers the marginal count near the observed 0.8 after integrating "
                            "over an approximately unit-scale static latent, with wide uncertainty "
                            "because only five family-history observations exist."
                        ),
                    },
                    "t0_sd_genetic_family_liability": {
                        "distribution": "LogNormal",
                        "params": {"mu": 0.0, "sigma": 0.3},
                        "reasoning": (
                            "Maintains the compiler's unit-scale convention for this time-invariant "
                            "trait while allowing moderate uncertainty in its initial spread."
                        ),
                    },
                },
            },
            {
                "construct": "early_life_adversity",
                "indicators": [
                    {
                        "variable": "early_life_adversity_history",
                        "family": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": (
                            "The four declared levels are ordered structural support. Only level 0 "
                            "is observed once, so the compatible ordinal likelihood is retained and "
                            "the resulting weak learning is treated as a data limitation."
                        ),
                    }
                ],
                "priors": {
                    "obs_ordered_base": {
                        "distribution": "Normal",
                        "params": {"mu": 0.0, "sigma": 1.0},
                        "reasoning": (
                            "Weakly centers the pooled ordered-logistic threshold bases while the "
                            "sparse ordinal channels contribute little empirical location information."
                        ),
                    },
                    "obs_ordered_gaps": {
                        "distribution": "HalfNormal",
                        "params": {"sigma": 1.0},
                        "reasoning": (
                            "Keeps adjacent pooled ordered-logistic thresholds separated without "
                            "claiming precise category spacing from the sparse static histories."
                        ),
                    },
                    "t0_sd_early_life_adversity": {
                        "distribution": "LogNormal",
                        "params": {"mu": 0.0, "sigma": 0.3},
                        "reasoning": (
                            "Matches the pooled static-state scale family and preserves an "
                            "approximately unit-scale latent despite extremely sparse measurement."
                        ),
                    },
                },
            },
            {
                "construct": "temperament_neuroticism_behavioral_inhibition",
                "indicators": [
                    {
                        "variable": "trait_negative_affectivity_level",
                        "family": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": (
                            "The declared low-to-very-high levels are ordered. Three observations "
                            "occupy the middle two categories, so an ordered-logistic emission "
                            "preserves that support without pretending the sparse occupancy "
                            "identifies category probabilities precisely."
                        ),
                    }
                ],
                "priors": {
                    "t0_sd_temperament_neuroticism_behavioral_inhibition": {
                        "distribution": "LogNormal",
                        "params": {"mu": 0.0, "sigma": 0.3},
                        "reasoning": (
                            "Matches the pooled static-state scale family and maintains an "
                            "approximately unit-scale trait under sparse measurement."
                        ),
                    },
                },
            },
            {
                "construct": "temperament_neuroticism_behavioral_inhibition",
                "indicators": [
                    {
                        "variable": "trait_negative_affectivity_level",
                        "family": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": (
                            "The ordered-logistic support remains correct; this revision moves "
                            "prior mass toward the two observed middle categories without changing "
                            "the declared four-level support."
                        ),
                    }
                ],
                "priors": {
                    "t0_sd_temperament_neuroticism_behavioral_inhibition": {
                        "distribution": "LogNormal",
                        "params": {"mu": -0.5, "sigma": 0.2},
                        "reasoning": (
                            "Narrows the static latent enough to favor middle categories while "
                            "remaining within the C2 standardized-latent scale band."
                        ),
                    },
                },
                "accept": [
                    {
                        "check": "C5a location reach",
                        "target": "trait_negative_affectivity_level",
                        "rationale": (
                            "Only three stable-trait observations exist and they occupy the two "
                            "middle categories. Calibrating a shared ordinal-threshold prior to "
                            "that empirical frequency would overstate the information in this "
                            "channel; retaining broad prior-predictive category uncertainty is the "
                            "more honest consequence."
                        ),
                    }
                ],
            },
            {
                "construct": "comorbid_psychiatric_vulnerability",
                "indicators": [
                    {
                        "variable": "comorbid_psychiatric_diagnosis_count",
                        "family": "poisson",
                        "link": "log",
                        "reasoning": (
                            "The 57 observations are nonnegative integer counts with variance close "
                            "to the mean and no empirical evidence that an extra dispersion parameter "
                            "is identifiable, so Poisson is the parsimonious compatible emission."
                        ),
                    }
                ],
                "priors": {
                    "manifest_mean_comorbid_psychiatric_diagnosis_count": {
                        "distribution": "Normal",
                        "params": {"mu": -0.9, "sigma": 0.5},
                        "reasoning": (
                            "Centers the marginal count near the observed 0.67 after log-link mean "
                            "inflation from an approximately unit-scale static latent, while leaving "
                            "room for sparse-history uncertainty."
                        ),
                    },
                    "t0_sd_comorbid_psychiatric_vulnerability": {
                        "distribution": "LogNormal",
                        "params": {"mu": 0.0, "sigma": 0.3},
                        "reasoning": (
                            "Matches the pooled static-state scale family and preserves the "
                            "standardized-latent convention for this time-invariant vulnerability."
                        ),
                    },
                },
            },
            {
                "construct": "stable_ssri_pharmacodynamic_responsiveness",
                "indicators": [
                    {
                        "variable": "ssri_responsiveness_evidence_level",
                        "family": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": (
                            "The four declared responsiveness levels are ordered and all but the "
                            "lowest appear in 68 observations, so ordered logistic preserves the "
                            "structural codebook without treating the scores as metric."
                        ),
                    }
                ],
                "priors": {
                    "t0_sd_stable_ssri_pharmacodynamic_responsiveness": {
                        "distribution": "LogNormal",
                        "params": {"mu": 0.0, "sigma": 0.3},
                        "reasoning": (
                            "Matches the pooled static-state scale family and retains an "
                            "approximately unit-scale responsiveness trait."
                        ),
                    },
                },
            },
            {
                "construct": "cyp2c19_pharmacokinetic_capacity",
                "indicators": [
                    {
                        "variable": "cyp2c19_genotype_metabolizer_evidence",
                        "family": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": (
                            "The poor-to-ultrarapid metabolizer codebook is intrinsically ordered, "
                            "and the 15 genotype-derived observations occupy four of its five "
                            "declared levels, so ordered logistic preserves the pharmacokinetic "
                            "ranking without treating category distances as metric."
                        ),
                    }
                ],
                "priors": {
                    "t0_sd_cyp2c19_pharmacokinetic_capacity": {
                        "distribution": "LogNormal",
                        "params": {"mu": 0.0, "sigma": 0.3},
                        "reasoning": (
                            "Matches the pooled static-state scale family and retains an "
                            "approximately unit-scale latent pharmacokinetic capacity."
                        ),
                    },
                },
            },
            {
                "construct": "cyp2c19_pharmacokinetic_capacity",
                "indicators": [
                    {
                        "variable": "cyp2c19_genotype_metabolizer_evidence",
                        "family": "ordered_logistic",
                        "link": "cumulative_logit",
                        "reasoning": (
                            "The poor-to-ultrarapid metabolizer codebook is intrinsically ordered, "
                            "and the repeated genotype-derived observations retain the same "
                            "ordered-logistic measurement semantics."
                        ),
                    }
                ],
                "priors": {
                    "t0_sd_cyp2c19_pharmacokinetic_capacity": {
                        "distribution": "LogNormal",
                        "params": {"mu": 0.0, "sigma": 0.3},
                        "reasoning": (
                            "Retains the pooled static-state family and unit-scale convention; "
                            "the repeated genotype classification does not justify shrinking the "
                            "latent pharmacokinetic scale to mimic category occupancy."
                        ),
                    },
                },
                "accept": [
                    {
                        "check": "C5a location reach",
                        "target": "cyp2c19_genotype_metabolizer_evidence",
                        "rationale": (
                            "The 15 rows repeatedly encode one subject's stable genotype and are not "
                            "15 independent pharmacokinetic measurements. Moving the pooled ordinal "
                            "threshold prior toward their duplicated intermediate category would "
                            "overstate the information and distort the other ordered channels, so "
                            "the resulting prior sensitivity is the honest consequence."
                        ),
                    }
                ],
            },
        ]
        if proposal["construct"] == "external_stressful_events"
    ]
    return


@app.cell
def authored_proposals():
    N_DRAWS = 16
    SEED = 20260720
    PROPOSALS = [
        {
            "construct": "external_stressful_events",
            "indicators": [
                {
                    "variable": "external_stressor_event_count",
                    "family": "poisson",
                    "link": "log",
                    "reasoning": (
                        "The channel is a nonnegative integer count. Its variance-to-mean "
                        "ratio is only 1.39, and the dynamic log-scale latent already induces "
                        "extra-Poisson variation, so a separate dispersion parameter is not "
                        "yet justified."
                    ),
                }
            ],
            "priors": {
                "manifest_mean_external_stressor_event_count": {
                    "distribution": "Normal",
                    "params": {"mu": -3.45, "sigma": 0.5},
                    "reasoning": (
                        "The observed mean is 0.053 events/day. Centering below log(0.053) "
                        "allows for marginal mean inflation when an approximately unit-scale "
                        "latent enters the Poisson log rate with its fixed loading."
                    ),
                },
                "rho_external_stressful_events": {
                    "distribution": "Beta",
                    "params": {"alpha": 2.0, "beta": 5.0},
                    "reasoning": (
                        "External stressful events are primarily short-lived daily shocks, "
                        "so persistence over the one-day model interval should usually be "
                        "low while retaining some probability of multi-day episodes."
                    ),
                },
                "sigma_external_stressful_events": {
                    "distribution": "LogNormal",
                    "params": {"mu": 0.45, "sigma": 0.3},
                    "reasoning": (
                        "For the proposed fast relaxation, a diffusion scale near 1.6 keeps "
                        "the stationary latent spread near the compiler's unit-scale target "
                        "while allowing uncertainty in shock magnitude."
                    ),
                },
            },
        },
        {
            "construct": "hormonal_state_changes",
            "indicators": [
                {
                    "variable": "hormonal_change_mention",
                    "family": "bernoulli",
                    "link": "logit",
                    "reasoning": (
                        "The declared channel records whether an active hormonal change is "
                        "explicitly present on a day, so its support is binary even though "
                        "this panel contains no usable observations."
                    ),
                }
            ],
            "priors": {
                "beta_sex_reproductive_context_hormonal_state_changes": {
                    "distribution": "Normal",
                    "params": {"mu": 0.3, "sigma": 0.3},
                    "reasoning": (
                        "More active reproductive context can raise the propensity for "
                        "short-run hormonal changes, but the known input is sparse and nearly "
                        "constant in this panel, so the effect is strongly regularized."
                    ),
                },
                "manifest_mean_hormonal_change_mention": {
                    "distribution": "Normal",
                    "params": {"mu": -2.5, "sigma": 1.0},
                    "reasoning": (
                        "In the absence of observed outcomes, this broad prior represents an "
                        "explicit hormonal-change mention as uncommon on any given day without "
                        "claiming a panel-derived prevalence."
                    ),
                },
                "rho_hormonal_state_changes": {
                    "distribution": "Beta",
                    "params": {"alpha": 9.0, "beta": 2.0},
                    "reasoning": (
                        "Reproductive or endocrine changes relevant to mood and sleep usually "
                        "persist across several days rather than appearing as isolated daily "
                        "shocks."
                    ),
                },
                "sigma_hormonal_state_changes": {
                    "distribution": "LogNormal",
                    "params": {"mu": -0.45, "sigma": 0.35},
                    "reasoning": (
                        "Pairs the slower persistence prior with diffusion near 0.64 so the "
                        "unobserved latent remains approximately unit scale without allowing "
                        "an unmeasured nuisance state to dominate the model."
                    ),
                },
            },
        },
        {
            "construct": "hormonal_state_changes",
            "indicators": [
                {
                    "variable": "hormonal_change_mention",
                    "family": "bernoulli",
                    "link": "logit",
                    "reasoning": (
                        "The declared channel records whether an active hormonal change is "
                        "explicitly present on a day, so its support is binary even though "
                        "this panel contains no usable observations."
                    ),
                }
            ],
            "priors": {
                "beta_sex_reproductive_context_hormonal_state_changes": {
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.1},
                    "reasoning": (
                        "The known input is sparse and nearly constant for this one person, "
                        "so its coefficient is not empirically separable from the latent "
                        "baseline. Strong zero-centered shrinkage prevents that unidentified "
                        "effect from inflating the prior-only state."
                    ),
                },
                "manifest_mean_hormonal_change_mention": {
                    "distribution": "Normal",
                    "params": {"mu": -2.5, "sigma": 1.0},
                    "reasoning": (
                        "In the absence of observed outcomes, this broad prior represents an "
                        "explicit hormonal-change mention as uncommon on any given day without "
                        "claiming a panel-derived prevalence."
                    ),
                },
                "rho_hormonal_state_changes": {
                    "distribution": "Beta",
                    "params": {"alpha": 9.0, "beta": 2.0},
                    "reasoning": (
                        "Reproductive or endocrine changes relevant to mood and sleep usually "
                        "persist across several days rather than appearing as isolated daily "
                        "shocks."
                    ),
                },
                "sigma_hormonal_state_changes": {
                    "distribution": "LogNormal",
                    "params": {"mu": -0.6, "sigma": 0.25},
                    "reasoning": (
                        "Regularizes the unobserved state below unit marginal innovation scale "
                        "while retaining the several-day persistence assumption; uncertainty "
                        "from this nuisance process should be propagated, not allowed to "
                        "dominate measured states."
                    ),
                },
            },
            "accept": [
                {
                    "check": "C3 resolvability",
                    "target": "hormonal_state_changes",
                    "rationale": (
                        "No direct hormonal observations exist, so no prior revision can "
                        "create temporal contrast. The several-day timescale is retained only "
                        "to propagate plausible nuisance variation into measured descendants; "
                        "it must not be reported as learned and requires posterior-contraction "
                        "and sensitivity checks before downstream numeric use."
                    ),
                },
                {
                    "check": "C5d data availability",
                    "target": "hormonal_change_mention",
                    "rationale": (
                        "The panel contains no usable hormone-specific values. Keeping the "
                        "declared Bernoulli channel makes the forward model explicit, but its "
                        "emission and linked trajectory remain prior-driven and must not be "
                        "presented as empirically learned."
                    ),
                },
            ],
        },
        {
            "construct": "internalizing_symptom_burden",
            "indicators": [
                {
                    "variable": "phq9_screening_score",
                    "family": "ordered_logistic",
                    "link": "cumulative_logit",
                    "reasoning": (
                        "PHQ-9 is a bounded ordered score. The five observed totals are sparse "
                        "but retain their ordering, so collapsing them to unordered categories "
                        "would discard defensible measurement information."
                    ),
                },
                {
                    "variable": "gad7_screening_score",
                    "family": "ordered_logistic",
                    "link": "cumulative_logit",
                    "reasoning": (
                        "GAD-7 is a bounded ordered score. Its five observations cannot support "
                        "a flexible unordered class model, while the cumulative-logit channel "
                        "preserves the declared score ordering."
                    ),
                },
                {
                    "variable": "state_of_mind_valence",
                    "family": "gaussian",
                    "link": "identity",
                    "reasoning": (
                        "Daily mean valence is continuous and dense relative to the screening "
                        "scores. The compiler standardizes this additive-location channel; its "
                        "fixed negative reference loading anchors higher latent burden to worse "
                        "valence."
                    ),
                },
                {
                    "variable": "journal_internalizing_symptom_severity",
                    "family": "ordered_logistic",
                    "link": "cumulative_logit",
                    "reasoning": (
                        "The journal-derived none/mild/moderate/severe codebook is intrinsically "
                        "ordered. Ordered logistic respects that support while the prior "
                        "predictive check exposes the sparse severe category."
                    ),
                },
            ],
            "priors": {
                "beta_hormonal_state_changes_internalizing_symptom_burden": {
                    "distribution": "Normal",
                    "params": {"mu": 0.0, "sigma": 0.1},
                    "reasoning": (
                        "The hormonal parent is entirely prior-driven in this panel. Strong "
                        "zero-centered shrinkage propagates plausible uncertainty without "
                        "letting an unobserved nuisance trajectory dominate measured symptoms."
                    ),
                },
                "beta_pre_taper_residual_symptom_burden_internalizing_symptom_burden": {
                    "distribution": "Normal",
                    "params": {"mu": 0.3, "sigma": 0.15},
                    "reasoning": (
                        "The observed baseline residual-symptom code spans 0 to 2 and should "
                        "positively anchor early internalizing burden, but repeated values for "
                        "one person warrant a regularized rather than large transition effect."
                    ),
                },
                "lambda_phq9_screening_score_internalizing_symptom_burden": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 1.0, "sigma": 0.35, "lower": 0.05, "upper": 3.0},
                    "reasoning": (
                        "Higher PHQ-9 scores indicate greater burden. A positive loading near "
                        "one keeps its cumulative-logit predictor commensurate with the "
                        "unit-scale latent while allowing moderate uncertainty."
                    ),
                },
                "lambda_gad7_screening_score_internalizing_symptom_burden": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 0.9, "sigma": 0.35, "lower": 0.05, "upper": 3.0},
                    "reasoning": (
                        "Higher GAD-7 scores indicate greater burden. The positive, regularized "
                        "loading allows anxiety severity to track the shared state without "
                        "claiming identical sensitivity to PHQ-9."
                    ),
                },
                "lambda_journal_internalizing_symptom_severity_internalizing_symptom_burden": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 0.8, "sigma": 0.35, "lower": 0.05, "upper": 3.0},
                    "reasoning": (
                        "Higher journal severity indicates greater burden, but the coarse "
                        "four-level rubric should have a somewhat more regularized loading than "
                        "the formal screening scales."
                    ),
                },
                "obs_ordered_base_phq9_screening_score": {
                    "distribution": "Normal",
                    "params": {"mu": -3.5, "sigma": 0.8},
                    "reasoning": (
                        "The five PHQ-9 totals occupy categories 3 through 16 of the declared "
                        "0--27 scale. With roughly half-logit adjacent spacing, a base near "
                        "-3.5 places the prior midpoint around the observed score range while "
                        "remaining broad enough for this sparse screening channel."
                    ),
                },
                "obs_ordered_gaps_phq9_screening_score": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.6},
                    "reasoning": (
                        "Half-logit-scale positive spacings spread the 27 PHQ-9 cutpoints across "
                        "the plausible predictor range without treating five observed totals as "
                        "enough information to estimate each threshold separately."
                    ),
                },
                "obs_ordered_base_gad7_screening_score": {
                    "distribution": "Normal",
                    "params": {"mu": -2.5, "sigma": 0.7},
                    "reasoning": (
                        "The five GAD-7 totals occupy categories 2 through 11 of the declared "
                        "0--21 scale. Its indicator-specific base is therefore less negative "
                        "than the PHQ-9 base, centering prior mass in the observed score range "
                        "without claiming precise threshold locations."
                    ),
                },
                "obs_ordered_gaps_gad7_screening_score": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.6},
                    "reasoning": (
                        "Positive spacings near half a logit distribute the 21 GAD-7 cutpoints "
                        "over a useful predictor range while retaining substantial uncertainty "
                        "from only five screening visits."
                    ),
                },
                "obs_ordered_base_journal_internalizing_symptom_severity": {
                    "distribution": "Normal",
                    "params": {"mu": -1.45, "sigma": 0.5},
                    "reasoning": (
                        "About 19% of journal days are coded none, so a first threshold near "
                        "logit(0.19) anchors the four-level rubric on its own empirical scale "
                        "rather than sharing a location with the much longer screening scales."
                    ),
                },
                "obs_ordered_gaps_journal_internalizing_symptom_severity": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 3.0},
                    "reasoning": (
                        "The journal channel is mostly mild, sometimes moderate, and never "
                        "severe in this panel. Its two cutpoints therefore need much wider "
                        "spacing than adjacent integer totals on PHQ-9 or GAD-7; the broad prior "
                        "keeps the unobserved severe category possible without making it common."
                    ),
                },
                "obs_sd_state_of_mind_valence": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "reasoning": (
                        "On the compiler-standardized valence scale, moderate residual noise "
                        "allows daily measurements to anchor the state without treating a single "
                        "self-report channel as nearly deterministic."
                    ),
                },
                "rho_internalizing_symptom_burden": {
                    "distribution": "Beta",
                    "params": {"alpha": 8.0, "beta": 2.0},
                    "reasoning": (
                        "Depressive and anxiety symptom burden usually persists over several "
                        "days, while the prior still allows meaningful week-scale movement."
                    ),
                },
                "sigma_internalizing_symptom_burden": {
                    "distribution": "LogNormal",
                    "params": {"mu": -0.4, "sigma": 0.25},
                    "reasoning": (
                        "With the several-day persistence prior, diffusion near 0.67 targets an "
                        "approximately unit-scale stationary symptom state and matches the "
                        "already established pooled diffusion family."
                    ),
                },
            },
        },
    ]
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "beta_pre_taper_residual_symptom_burden_internalizing_symptom_burden": {
                    "distribution": "Normal",
                    "params": {"mu": 0.15, "sigma": 0.1},
                    "reasoning": (
                        "The first exact pass showed that the baseline input could move the "
                        "child more than its own temporal path. Halving and tightening the "
                        "effect retains a positive early-trajectory anchor without allowing "
                        "three repeated baseline categories to govern the dynamic state."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "beta_pre_taper_residual_symptom_burden_internalizing_symptom_burden": {
                    "distribution": "Normal",
                    "params": {"mu": 0.24, "sigma": 0.12},
                    "reasoning": (
                        "The halved second-pass effect cleared edge overwhelm but left too "
                        "little child movement for the sparse PHQ-9 and GAD-7 channels to carry "
                        "the minimum temporal signal share. A moderate effect between the first "
                        "two proposals preserves the baseline trajectory anchor while remaining "
                        "below the first pass that dominated the child's own temporal path."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "beta_pre_taper_residual_symptom_burden_internalizing_symptom_burden": {
                    "distribution": "Normal",
                    "params": {"mu": 0.15, "sigma": 0.1},
                    "reasoning": (
                        "The moderate third-pass effect restored screening transmission but "
                        "still displaced more of the child path than its own temporal dynamics. "
                        "Return to the non-dominating second-pass effect and address weak "
                        "measurement transmission on the screening emission surfaces."
                    ),
                },
                "lambda_phq9_screening_score_internalizing_symptom_burden": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 1.2, "sigma": 0.3, "lower": 0.05, "upper": 3.0},
                    "reasoning": (
                        "PHQ-9 is a formal multi-item symptom scale and should respond strongly "
                        "to the shared internalizing state. The second exact pass put its "
                        "temporal signal share at the admission boundary, so a modestly stronger "
                        "and tighter positive loading is preferable to inflating a structural "
                        "input effect."
                    ),
                },
                "lambda_gad7_screening_score_internalizing_symptom_burden": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 1.15, "sigma": 0.3, "lower": 0.05, "upper": 3.0},
                    "reasoning": (
                        "GAD-7 is likewise a direct multi-item measure of internalizing burden. "
                        "Its second-pass temporal signal share was just below the minimum, so "
                        "the loading is strengthened without claiming deterministic agreement "
                        "with the latent state."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            "construct": "depressive_anxiety_disorder_activity",
            "indicators": [
                {
                    "variable": "clinical_disorder_activity_assessment",
                    "family": "ordered_logistic",
                    "link": "cumulative_logit",
                    "reasoning": (
                        "The clinician assessment codebook progresses from remission through "
                        "mild, moderate, and severe active illness. Ordered logistic preserves "
                        "that ranking and the reference loading fixes higher latent disorder "
                        "activity to higher assessed severity."
                    ),
                }
            ],
            "priors": {
                "beta_depressive_anxiety_disorder_activity_internalizing_symptom_burden": {
                    "distribution": "Normal",
                    "params": {"mu": 0.12, "sigma": 0.06},
                    "reasoning": (
                        "Greater active depressive or anxiety illness should increase the "
                        "downstream symptom-burden state. The loop-closing effect is kept "
                        "moderate so it transmits clinically meaningful recurrence without "
                        "overriding the child's own dynamics."
                    ),
                },
                "beta_natural_recovery_propensity_depressive_anxiety_disorder_activity": {
                    "distribution": "Normal",
                    "params": {"mu": -0.03, "sigma": 0.025},
                    "reasoning": (
                        "Natural remission and resilience should suppress active illness. The "
                        "input is sparse, binary in this person, and forward-filled, so the "
                        "daily effect is strongly regularized rather than treated as repeatedly "
                        "measured independent evidence."
                    ),
                },
                "beta_pre_taper_remission_stability_depressive_anxiety_disorder_activity": {
                    "distribution": "Normal",
                    "params": {"mu": -0.02, "sigma": 0.015},
                    "reasoning": (
                        "More stable pre-taper remission should lower near-term disorder "
                        "activity. Because the observed input spans 0--3 and can persist between "
                        "assessments, a small daily coefficient prevents stable baseline context "
                        "from dominating subsequent illness dynamics."
                    ),
                },
                "beta_relapse_vulnerability_depressive_anxiety_disorder_activity": {
                    "distribution": "Normal",
                    "params": {"mu": 0.03, "sigma": 0.025},
                    "reasoning": (
                        "Documented relapse vulnerability should raise underlying illness "
                        "activity. Its two observed levels and forward-filled use support only "
                        "a small positive daily effect with substantial mass near zero."
                    ),
                },
                "obs_ordered_base_clinical_disorder_activity_assessment": {
                    "distribution": "Normal",
                    "params": {"mu": -0.4, "sigma": 0.5},
                    "reasoning": (
                        "About 41% of assessments are none or remitted, so the first threshold "
                        "is centered near logit(0.41) on the reference predictor scale while "
                        "remaining broad enough for duplicated and irregular clinical notes."
                    ),
                },
                "obs_ordered_gaps_clinical_disorder_activity_assessment": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 2.5},
                    "reasoning": (
                        "Mild assessments are common, moderate assessments less common, and "
                        "severe activity is unobserved. Broad positive gaps around two logits "
                        "represent that separation while preserving nonzero prior probability "
                        "for severe recurrence."
                    ),
                },
                "rho_depressive_anxiety_disorder_activity": {
                    "distribution": "Beta",
                    "params": {"alpha": 29.0, "beta": 1.0},
                    "reasoning": (
                        "Underlying depressive or anxiety episodes evolve over weeks rather "
                        "than resetting daily. Daily persistence near 0.97 gives an approximately "
                        "month-scale baseline settling time that is resolvable by the 79 "
                        "irregular clinical assessments over seven years."
                    ),
                },
                "sigma_depressive_anxiety_disorder_activity": {
                    "distribution": "LogNormal",
                    "params": {"mu": -1.3, "sigma": 0.3},
                    "reasoning": (
                        "With month-scale persistence, diffusion near 0.27 targets roughly "
                        "unit-scale stationary illness variation before incoming drivers. The "
                        "multiplicative uncertainty permits quieter remission periods and more "
                        "volatile recurrence without inviting unbounded trajectories."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "sigma_depressive_anxiety_disorder_activity": {
                    "distribution": "LogNormal",
                    "params": {"mu": -1.7, "sigma": 0.25},
                    "reasoning": (
                        "The first exact pass was confined and resolvable but placed median "
                        "latent scale at 3.16, just above the standardized-state band. Lowering "
                        "and tightening diffusion targets a unit-to-moderate illness scale while "
                        "preserving the month-scale persistence and measured input effects that "
                        "already passed their reachability checks."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "rho_depressive_anxiety_disorder_activity": {
                    "distribution": "Beta",
                    "params": {"alpha": 24.0, "beta": 1.2},
                    "reasoning": (
                        "The lower-diffusion second pass still produced excessive scale, while "
                        "the stable-input displacement ratios increased. This implicates the "
                        "roughly two-month settling time rather than stochastic innovation. A "
                        "more regularized persistence prior targets illness adjustment over "
                        "several weeks, still longer than transient symptoms and well resolved "
                        "by the clinical assessment span."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            "construct": "escitalopram_dose_taken",
            "indicators": [
                {
                    "variable": "escitalopram_documented_dose_mg",
                    "family": "gaussian",
                    "link": "identity",
                    "reasoning": (
                        "Documented milligram dose is a continuous amount even though this "
                        "person's records cluster at prescribed step levels. The additive "
                        "last-value channel is compiler-standardized and serves as the positive "
                        "reference measure of actual dose taken."
                    ),
                },
                {
                    "variable": "escitalopram_fill_quantity",
                    "family": "gaussian",
                    "link": "identity",
                    "reasoning": (
                        "Fill quantity is a highly noisy raw-scale availability proxy rather "
                        "than a daily ingestion count: fills occur in bundles and zero means no "
                        "fill event that day, not zero medication available. A broad Gaussian "
                        "channel represents that weak linear association without imposing a "
                        "count-process interpretation."
                    ),
                },
            ],
            "priors": {
                "beta_baseline_maintenance_dose_level_escitalopram_dose_taken": {
                    "distribution": "Normal",
                    "params": {"mu": 0.005, "sigma": 0.003},
                    "reasoning": (
                        "Baseline prescribed dose should positively anchor actual dose, but the "
                        "input remains on its 5--20 mg scale and is forward-filled. A small daily "
                        "coefficient lets those levels shift a standardized latent state without "
                        "allowing a persistent prescription field to determine it."
                    ),
                },
                "beta_counterfactual_taper_regime_escitalopram_dose_taken": {
                    "distribution": "Normal",
                    "params": {"mu": -0.03, "sigma": 0.02},
                    "reasoning": (
                        "Assignment to taper-off should lower subsequent dose relative to "
                        "continuation, while most mechanical reduction is reserved for the "
                        "deferred taper-speed parent. The regularized direct effect represents "
                        "the regime constraint without forcing an instantaneous dose collapse."
                    ),
                },
                "lambda_escitalopram_fill_quantity_escitalopram_dose_taken": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 10.0, "sigma": 5.0, "lower": 0.1, "upper": 30.0},
                    "reasoning": (
                        "A unit change in the standardized dose state can plausibly correspond "
                        "to roughly ten dispensed units, but pharmacy bundle sizes and timing "
                        "make this a weak proxy. The broad positive loading preserves direction "
                        "without equating a fill with ingestion."
                    ),
                },
                "manifest_mean_escitalopram_fill_quantity": {
                    "distribution": "Normal",
                    "params": {"mu": 52.0, "sigma": 15.0},
                    "reasoning": (
                        "The raw sum channel is not compiler-standardized. Its intercept is "
                        "therefore centered near the observed mean fill quantity, with broad "
                        "uncertainty for the sparse mixture of 0-, 30-, and 90-unit records."
                    ),
                },
                "obs_sd_escitalopram_documented_dose_mg": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.5},
                    "reasoning": (
                        "On the compiler-standardized reference scale, moderate residual noise "
                        "acknowledges that a documented prescribed dose is only a proxy for the "
                        "amount actually ingested."
                    ),
                },
                "obs_sd_escitalopram_fill_quantity": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 40.0},
                    "reasoning": (
                        "Raw fill quantities vary by nearly 39 units and occur in coarse bundles. "
                        "A large residual scale prevents sparse pharmacy events from acting like "
                        "precise repeated measurements of daily dose."
                    ),
                },
                "rho_escitalopram_dose_taken": {
                    "distribution": "Beta",
                    "params": {"alpha": 24.0, "beta": 1.2},
                    "reasoning": (
                        "Actual daily dose stays near a prescribed step until a taper or "
                        "adherence change occurs. Several-week persistence is consistent with "
                        "that stepwise process and resolvable by the documented-dose schedule."
                    ),
                },
                "sigma_escitalopram_dose_taken": {
                    "distribution": "LogNormal",
                    "params": {"mu": -1.5, "sigma": 0.25},
                    "reasoning": (
                        "With several-week persistence, diffusion near 0.22 permits modest "
                        "day-to-day ingestion departures while keeping prescription level and "
                        "regime as the main sources of sustained movement."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "beta_baseline_maintenance_dose_level_escitalopram_dose_taken": {
                    "distribution": "Normal",
                    "params": {"mu": 0.002, "sigma": 0.0015},
                    "reasoning": (
                        "The first exact pass placed latent scale at 4.48 and the unscaled "
                        "5--20 mg baseline input displaced 91.9% of the child path. Shrinking "
                        "this daily effect preserves prescription anchoring while allowing "
                        "dose dynamics and adherence-related innovation to remain identifiable."
                    ),
                },
                "lambda_escitalopram_fill_quantity_escitalopram_dose_taken": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 8.0, "sigma": 4.0, "lower": 0.1, "upper": 25.0},
                    "reasoning": (
                        "The first pass coupled a high latent level with a ten-unit loading and "
                        "overpredicted fill location. A slightly smaller but still positive "
                        "loading reflects pharmacy availability as a weak dose proxy rather "
                        "than a direct milligram measurement."
                    ),
                },
                "manifest_mean_escitalopram_fill_quantity": {
                    "distribution": "Normal",
                    "params": {"mu": 30.0, "sigma": 10.0},
                    "reasoning": (
                        "Admission compares the sparse fill channel through its median, which "
                        "is 30 units despite a mean of 52 from occasional 90-unit fills. "
                        "Centering the raw-scale intercept at that typical bundle avoids letting "
                        "the right-skewed mean define every prior replicate."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "self_limit_escitalopram_dose_taken": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.02},
                    "reasoning": (
                        "The second exact pass satisfied scale and emission checks but two of "
                        "sixteen high-persistence draws grew without settling. Actual ingested "
                        "dose has a hard practical range, so a weak quartic restoring term is "
                        "scientifically preferable to erasing legitimate step persistence; it "
                        "is negligible near the center and strengthens only on large excursions."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            "construct": "taper_speed_dose_reduction",
            "indicators": [
                {
                    "variable": "escitalopram_taper_action_flag",
                    "family": "bernoulli",
                    "link": "logit",
                    "reasoning": (
                        "The action flag is binary by construction. A logit channel preserves "
                        "the distinction between explicitly continuing and explicitly reducing "
                        "escitalopram while allowing the three observed zeros to remain weak "
                        "rather than definitive evidence against future taper action."
                    ),
                },
                {
                    "variable": "taper_speed_instruction_intensity",
                    "family": "ordered_logistic",
                    "link": "cumulative_logit",
                    "reasoning": (
                        "The instruction rubric is ordered from continuation through slow, "
                        "moderate, and rapid reduction. Ordered logistic preserves increasing "
                        "taper intensity, with the fixed positive reference loading anchoring "
                        "the latent direction."
                    ),
                },
            ],
            "priors": {
                "beta_counterfactual_taper_regime_taper_speed_dose_reduction": {
                    "distribution": "Normal",
                    "params": {"mu": 0.12, "sigma": 0.06},
                    "reasoning": (
                        "Assignment to taper-off should directly raise attempted reduction "
                        "speed relative to continuation. The effect is positive but regularized "
                        "because clinician guidance and patient response, authored later, also "
                        "determine the realized plan."
                    ),
                },
                "beta_medication_formulation_constraints_taper_speed_dose_reduction": {
                    "distribution": "Normal",
                    "params": {"mu": -0.03, "sigma": 0.02},
                    "reasoning": (
                        "The source code increases from standard tablets toward flexible small "
                        "steps and liquid formulations, so higher values should permit slower "
                        "reductions. Its forward-filled 1--3 scale receives a small negative "
                        "daily effect."
                    ),
                },
                "beta_taper_speed_dose_reduction_escitalopram_dose_taken": {
                    "distribution": "Normal",
                    "params": {"mu": -0.08, "sigma": 0.04},
                    "reasoning": (
                        "Faster or larger planned reductions mechanically lower subsequent "
                        "actual dose. The loop-closing effect is moderate so a taper plan changes "
                        "dose without overpowering baseline prescription, adherence, and the "
                        "dose state's own bounded dynamics."
                    ),
                },
                "lambda_escitalopram_taper_action_flag_taper_speed_dose_reduction": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 0.8, "sigma": 0.35, "lower": 0.05, "upper": 3.0},
                    "reasoning": (
                        "Explicit taper action should become more likely as latent reduction "
                        "speed rises, but the flag is based on only three regimen-discussion "
                        "days and therefore receives a regularized positive loading."
                    ),
                },
                "manifest_mean_escitalopram_taper_action_flag": {
                    "distribution": "Normal",
                    "params": {"mu": -3.0, "sigma": 1.0},
                    "reasoning": (
                        "All three observed action flags are zero. A low baseline log-odds prior "
                        "makes explicit taper action uncommon during maintenance while retaining "
                        "meaningful probability under a positive taper state."
                    ),
                },
                "obs_ordered_base_taper_speed_instruction_intensity": {
                    "distribution": "Normal",
                    "params": {"mu": 0.45, "sigma": 0.5},
                    "reasoning": (
                        "About 61% of documented instructions are continuation or no reduction, "
                        "placing the first cumulative threshold near logit(0.61) on the reference "
                        "predictor scale."
                    ),
                },
                "obs_ordered_gaps_taper_speed_instruction_intensity": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 2.5},
                    "reasoning": (
                        "Slow reductions account for most remaining observations, moderate "
                        "reductions are uncommon, and rapid stops are absent. Broad positive "
                        "gaps around two logits represent that ordered separation without "
                        "excluding a future rapid taper."
                    ),
                },
                "rho_taper_speed_dose_reduction": {
                    "distribution": "Beta",
                    "params": {"alpha": 18.0, "beta": 2.0},
                    "reasoning": (
                        "A taper instruction or pause generally persists for days to weeks, not "
                        "just one day, but speed can change at the next clinical adjustment. "
                        "This prior targets that intermediate settling time."
                    ),
                },
                "sigma_taper_speed_dose_reduction": {
                    "distribution": "LogNormal",
                    "params": {"mu": -0.8, "sigma": 0.25},
                    "reasoning": (
                        "Moderate innovation permits unrecorded day-to-day plan adjustments "
                        "while the regime, formulation, and persistent taper state supply most "
                        "of the structured variation."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "lambda_escitalopram_taper_action_flag_taper_speed_dose_reduction": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": 1.5, "sigma": 0.5, "lower": 0.05, "upper": 3.0},
                    "reasoning": (
                        "The first exact pass put only 1.5% temporal signal into the sparse "
                        "action flag. Explicit action language should be strongly discriminating "
                        "when it appears, so the loading is increased while remaining uncertain "
                        "because only three zero-valued observations are available."
                    ),
                },
                "manifest_mean_escitalopram_taper_action_flag": {
                    "distribution": "Normal",
                    "params": {"mu": -2.5, "sigma": 0.8},
                    "reasoning": (
                        "The first pass located the three observed zeros but saturated the "
                        "logit channel. A still-low but less extreme baseline keeps taper action "
                        "uncommon during maintenance while allowing latent taper speed to move "
                        "its probability."
                    ),
                },
                "rho_taper_speed_dose_reduction": {
                    "distribution": "Beta",
                    "params": {"alpha": 24.0, "beta": 1.5},
                    "reasoning": (
                        "The first prior relaxed in a median 13.7 days, faster than the 19-day "
                        "median instruction gap, leaving only 75% resolvable. Several-week "
                        "persistence better matches how taper instructions remain active until "
                        "a subsequent adjustment."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            "construct": "adherence_to_regimen",
            "indicators": [
                {
                    "variable": "medication_adherence_documented",
                    "family": "ordered_logistic",
                    "link": "cumulative_logit",
                    "reasoning": (
                        "The documented adherence rubric is ordered from poor through partial "
                        "to consistent. Ordered logistic preserves that direction and the fixed "
                        "positive reference loading anchors higher latent adherence to more "
                        "consistent observed use."
                    ),
                },
                {
                    "variable": "missed_or_extra_escitalopram_dose_mentions",
                    "family": "poisson",
                    "link": "log",
                    "reasoning": (
                        "This indicator is a nonnegative event count with mean and variance both "
                        "near 0.03. Poisson is adequate for the observed one-event maximum, and "
                        "the locked negative loading makes deviations less likely as adherence "
                        "improves."
                    ),
                },
            ],
            "priors": {
                "beta_adherence_to_regimen_escitalopram_dose_taken": {
                    "distribution": "Normal",
                    "params": {"mu": 0.04, "sigma": 0.025},
                    "reasoning": (
                        "More consistent adherence should raise actual dose relative to missed "
                        "or partial use. The loop-closing effect is modest because prescribed "
                        "baseline and taper speed already explain much of the dose trajectory."
                    ),
                },
                "beta_internalizing_symptom_burden_adherence_to_regimen": {
                    "distribution": "Normal",
                    "params": {"mu": -0.08, "sigma": 0.04},
                    "reasoning": (
                        "Higher depression or anxiety burden can impair organization and "
                        "motivation for following the intended regimen. The effect is negative "
                        "but regularized because symptoms can also prompt rescue adherence in "
                        "some periods."
                    ),
                },
                "lambda_missed_or_extra_escitalopram_dose_mentions_adherence_to_regimen": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": -1.0, "sigma": 0.4, "lower": -3.0, "upper": -0.05},
                    "reasoning": (
                        "Missed or extra-dose mentions are inverse indicators of adherence. A "
                        "one-unit latent increase should materially reduce their already-low log "
                        "rate, while the broad negative loading reflects only two nonzero events."
                    ),
                },
                "manifest_mean_missed_or_extra_escitalopram_dose_mentions": {
                    "distribution": "Normal",
                    "params": {"mu": -3.3, "sigma": 0.8},
                    "reasoning": (
                        "The observed rate is 0.032 mentions per documented adherence day, so "
                        "the Poisson log-rate intercept is centered near log(0.032) with broad "
                        "uncertainty for this rare channel."
                    ),
                },
                "obs_ordered_base_medication_adherence_documented": {
                    "distribution": "Normal",
                    "params": {"mu": -4.5, "sigma": 0.8},
                    "reasoning": (
                        "No observations are poor adherence, so the first threshold is placed "
                        "well below the central predictor while retaining prior mass for future "
                        "poor-adherence episodes."
                    ),
                },
                "obs_ordered_gaps_medication_adherence_documented": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 2.0},
                    "reasoning": (
                        "Only three observations are partial and 58 are consistent. A broad "
                        "positive gap places the second threshold near the low tail needed for "
                        "that imbalance without fixing either sparse category probability."
                    ),
                },
                "rho_adherence_to_regimen": {
                    "distribution": "Beta",
                    "params": {"alpha": 24.0, "beta": 1.5},
                    "reasoning": (
                        "Adherence habits persist over several weeks but can change after a new "
                        "instruction, symptom flare, or missed dose. This timescale matches the "
                        "roughly three-week observation cadence."
                    ),
                },
                "sigma_adherence_to_regimen": {
                    "distribution": "LogNormal",
                    "params": {"mu": -1.0, "sigma": 0.25},
                    "reasoning": (
                        "Moderate innovation allows intermittent deviations from an otherwise "
                        "stable adherence pattern while preserving a roughly standardized "
                        "latent scale under several-week persistence."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "beta_internalizing_symptom_burden_adherence_to_regimen": {
                    "distribution": "Normal",
                    "params": {"mu": -0.04, "sigma": 0.03},
                    "reasoning": (
                        "The first exact pass placed the symptom edge at 93% of child variation "
                        "and inflated adherence scale above the standardized band. Halving the "
                        "mean effect retains the expected negative relationship without letting "
                        "symptom dynamics define adherence."
                    ),
                },
                "lambda_missed_or_extra_escitalopram_dose_mentions_adherence_to_regimen": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": -0.6, "sigma": 0.3, "lower": -2.0, "upper": -0.05},
                    "reasoning": (
                        "The first pass produced an implausibly vast Poisson replicate envelope "
                        "because the log link exponentiated low-adherence tail draws. A smaller "
                        "negative loading still makes deviations less likely with better "
                        "adherence while respecting that only two events identify this channel."
                    ),
                },
                "sigma_adherence_to_regimen": {
                    "distribution": "LogNormal",
                    "params": {"mu": -1.2, "sigma": 0.25},
                    "reasoning": (
                        "Reducing innovation alongside the dominant symptom edge targets a "
                        "unit-to-moderate adherence state while retaining intermittent "
                        "day-to-day deviations from the persistent regimen pattern."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "priors": {
                **PROPOSALS[-1]["priors"],
                "lambda_missed_or_extra_escitalopram_dose_mentions_adherence_to_regimen": {
                    "distribution": "TruncatedNormal",
                    "params": {"mu": -0.75, "sigma": 0.2, "lower": -1.2, "upper": -0.1},
                    "reasoning": (
                        "After correcting latent scale, the rare count carried 3.8% temporal "
                        "signal, just below the 4% floor. A slightly stronger and much tighter "
                        "negative loading preserves the inverse adherence meaning while "
                        "excluding the extreme slopes that previously generated enormous "
                        "Poisson tail replicates."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            "construct": "plasma_escitalopram_exposure",
            "indicators": [
                {
                    "variable": "escitalopram_dose_exposure_proxy_mg",
                    "family": "gaussian",
                    "link": "identity",
                    "reasoning": (
                        "No blood concentration is available, so documented dose is an "
                        "imperfect continuous exposure proxy. The compiler-standardized "
                        "Gaussian reference channel preserves its direction while explicitly "
                        "leaving pharmacokinetic variability in the latent dynamics."
                    ),
                }
            ],
            "priors": {
                "beta_age_life_stage_plasma_escitalopram_exposure": {
                    "distribution": "Normal",
                    "params": {"mu": 0.03, "sigma": 0.025},
                    "reasoning": (
                        "Later life stage can modestly increase exposure through slower "
                        "clearance and comorbidity, but this person's source has only two levels "
                        "and is forward-filled, so the daily effect remains strongly regularized."
                    ),
                },
                "beta_baseline_physical_health_burden_plasma_escitalopram_exposure": {
                    "distribution": "Normal",
                    "params": {"mu": 0.04, "sigma": 0.03},
                    "reasoning": (
                        "Greater hepatic, renal, or general medical burden may increase systemic "
                        "exposure at a given dose. Only one moderate observation exists, so the "
                        "positive effect has substantial mass near zero."
                    ),
                },
                "beta_cyp2c19_pharmacokinetic_capacity_plasma_escitalopram_exposure": {
                    "distribution": "Normal",
                    "params": {"mu": -0.08, "sigma": 0.04},
                    "reasoning": (
                        "Higher CYP2C19 capacity denotes faster metabolism and therefore lower "
                        "escitalopram exposure at a fixed dose. The stable genotype-derived input "
                        "receives a regularized negative effect rather than repeated-data weight."
                    ),
                },
                "beta_escitalopram_dose_taken_plasma_escitalopram_exposure": {
                    "distribution": "Normal",
                    "params": {"mu": 0.3, "sigma": 0.12},
                    "reasoning": (
                        "Actual ingested dose is the proximal positive driver of systemic "
                        "exposure. A substantial but uncertain daily effect reflects roughly "
                        "linear therapeutic-range pharmacokinetics without claiming that the "
                        "dose proxy directly measures concentration."
                    ),
                },
                "rho_plasma_escitalopram_exposure": {
                    "distribution": "Beta",
                    "params": {"alpha": 12.0, "beta": 12.0},
                    "reasoning": (
                        "Escitalopram's roughly day-scale elimination implies exposure adjusts "
                        "within about one to two days after a dose change. This physical "
                        "timescale is intentionally much faster than the sparse documented-dose "
                        "schedule and must remain prior-driven in this panel."
                    ),
                },
                "sigma_plasma_escitalopram_exposure": {
                    "distribution": "LogNormal",
                    "params": {"mu": -0.2, "sigma": 0.25},
                    "reasoning": (
                        "With rapid clearance, innovation near 0.8 maintains an approximately "
                        "unit-scale exposure state while propagating unmeasured absorption, "
                        "timing, metabolism, and adherence variability."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "accept": [
                {
                    "check": "C3 resolvability",
                    "target": "plasma_escitalopram_exposure",
                    "rationale": (
                        "Escitalopram exposure physically equilibrates over roughly one to two "
                        "days, and slowing that process to match 37-day documentation gaps would "
                        "misrepresent pharmacokinetics. The panel has no blood concentrations, "
                        "so this timescale is retained solely as a prior-driven propagation "
                        "mechanism; it must not be described as learned and requires sensitivity "
                        "analysis in any downstream dose-response claim."
                    ),
                }
            ],
        }
    )
    PROPOSALS.append(
        {
            "construct": "serotonin_transporter_occupancy",
            "indicators": [
                {
                    "variable": "dose_based_sert_occupancy_proxy",
                    "family": "gaussian",
                    "link": "identity",
                    "reasoning": (
                        "No PET occupancy measure exists, so documented dose is only a monotonic "
                        "continuous proxy. The compiler-standardized Gaussian reference channel "
                        "anchors direction while the saturating exposure edge encodes the actual "
                        "nonlinear occupancy mechanism."
                    ),
                }
            ],
            "priors": {
                "hill_emax_plasma_escitalopram_exposure_serotonin_transporter_occupancy": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 0.8},
                    "reasoning": (
                        "Increasing plasma exposure raises transporter blockade but only up to a "
                        "finite effect. A moderate positive Emax allows material occupancy change "
                        "without an unbounded linear response."
                    ),
                },
                "hill_ec50_plasma_escitalopram_exposure_serotonin_transporter_occupancy": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 1.0},
                    "reasoning": (
                        "On the standardized exposure scale, half-maximal response is expected "
                        "within roughly one latent unit, with broad uncertainty because no blood "
                        "concentration or PET observations identify this threshold."
                    ),
                },
                "hill_n_plasma_escitalopram_exposure_serotonin_transporter_occupancy": {
                    "distribution": "HalfNormal",
                    "params": {"sigma": 2.0},
                    "reasoning": (
                        "A positive Hill exponent permits a gradual-to-moderately steep "
                        "therapeutic-range occupancy curve without fixing a precise binding "
                        "shape from the dose proxy."
                    ),
                },
                "rho_serotonin_transporter_occupancy": {
                    "distribution": "Beta",
                    "params": {"alpha": 10.0, "beta": 10.0},
                    "reasoning": (
                        "Transporter blockade tracks changing exposure over approximately a day "
                        "or two rather than persisting for weeks independently. This rapid "
                        "timescale is pharmacologically set and not learnable from sparse proxy "
                        "observations."
                    ),
                },
                "sigma_serotonin_transporter_occupancy": {
                    "distribution": "LogNormal",
                    "params": {"mu": -0.3, "sigma": 0.25},
                    "reasoning": (
                        "Innovation near 0.74 maintains a roughly standardized occupancy state "
                        "under rapid relaxation while allowing exposure timing and unmeasured "
                        "binding variability around the saturating response."
                    ),
                },
            },
        }
    )
    PROPOSALS.append(
        {
            **PROPOSALS[-1],
            "accept": [
                {
                    "check": "C3 resolvability",
                    "target": "serotonin_transporter_occupancy",
                    "rationale": (
                        "Transporter blockade follows plasma exposure over days, not the "
                        "37-day spacing of its dose proxy. Stretching occupancy kinetics to "
                        "match documentation would be pharmacologically false. The Hill response "
                        "and relaxation time are therefore prior-driven mechanisms only; binding "
                        "kinetics cannot be reported as learned and require sensitivity analysis "
                        "for downstream maintenance-effect claims."
                    ),
                }
            ],
        }
    )
    return N_DRAWS, PROPOSALS, SEED


@app.cell
def replay_proposals(
    N_DRAWS,
    PROPOSALS,
    SEED,
    WORKSPACE_ID,
    structural_plan,
    data_for_model,
    ps,
):
    workbench_run = ps.run_authored_proposals(
        cache_workspace_id=WORKSPACE_ID,
        structural_plan=structural_plan,
        data_for_model=data_for_model,
        proposals=PROPOSALS,
        n_draws=N_DRAWS,
        seed=SEED,
    )
    return (workbench_run,)


@app.cell
def attempt_reports(cs, mo, workbench_run):
    _panels = []
    for _attempt in workbench_run.attempts:
        _cache_label = " · cached" if _attempt.cache_hit else ""
        _title = f"{_attempt.construct} — authored attempt {_attempt.attempt}{_cache_label}"
        if _attempt.report is not None:
            _panels.append(cs.render_report(_title, _attempt.report))
        else:
            _kind = f" ({_attempt.error_type})" if _attempt.error_type else ""
            _panels.append(
                mo.md(
                    f"### {_title}\n\n`submit_construct` returned{_kind}:\n\n```\n{_attempt.feedback}\n```"
                )
            )
    mo.vstack(_panels) if _panels else mo.md("No proposals have been authored yet.")
    return


@app.cell
def next_authoring_prompt(
    structural_plan,
    mo,
    ps,
    question,
    validation_report,
    workbench_run,
):
    _prompt = ps.next_construct_prompt(
        run=workbench_run,
        question=question,
        structural_plan=structural_plan,
        validation_report=validation_report,
    )
    if _prompt is None:
        _display = mo.md(
            "## Authoring complete\n\nEvery construct is locally admitted; run the barrier below."
        )
    else:
        _system, _user = _prompt
        _display = mo.accordion(
            {
                f"Next production prompt: {workbench_run.state.current_construct}": mo.md(_user),
                "System guidance": mo.md(_system),
            }
        )
    mo.vstack([_display])
    return


@app.cell
def next_indicator_audits(
    structural_plan,
    get_manifest_indicators,
    indicator_audits,
    json,
    mo,
    workbench_run,
):
    _construct = workbench_run.state.current_construct
    _names = [
        indicator["name"]
        for indicator in get_manifest_indicators(structural_plan)
        if indicator.get("construct_name") == _construct
    ]
    _items = {
        f"Raw audit: {name}": mo.md(
            "```json\n" + json.dumps(indicator_audits[name], indent=2) + "\n```"
        )
        for name in _names
    }
    _display = (
        mo.accordion(_items)
        if _items
        else mo.md("No next-construct indicator audit: authoring is complete.")
    )
    mo.vstack([mo.md("### Source audit used to cross-check the rendered local context"), _display])
    return


@app.cell(hide_code=True)
def new_trial_issue_ledger(mo):
    mo.md(r"""
    ## New-trial framework issue ledger

    The prior replay's exact ordered-logistic location ridge is repaired in this run. Threshold and
    categorical channels no longer activate `manifest_mean_*`; auto-standardized continuous channels
    follow the same location-anchor rule. A second framework restriction surfaced when the
    internalizing construct combined 28-level PHQ-9, 22-level GAD-7, and four-level journal severity:
    the prompt exposed one global `obs_ordered_base` and `obs_ordered_gaps` prior for all three
    channels. That made the channel-specific category locations mutually incompatible. The
    production surface now exposes `obs_ordered_base_<indicator>` and
    `obs_ordered_gaps_<indicator>`. These bind to the relevant component or row of the existing
    vectorized runtime sites, so each indicator can express its own threshold geometry without
    changing the sampling topology.

    The stored DEMO snapshot's empty `known_inputs` declaration remains a documented upstream
    defect. This trial explicitly compiles its observed time-invariant quantities—including
    CYP2C19 capacity—as known inputs, so static-target chains are reduced before SSM lowering and
    every retained structural item receives a source-ID binding in the compiled artifact.

    The first inference smoke on the nine admitted constructs exposed two runtime bugs and one
    unresolved capability gap. First, fit preflight read hydrated descriptor defaults instead of
    the compiled prior runtime actually sampled by inference; it therefore rejected the authored
    raw-scale fill-quantity intercept. Preflight now reads the compiled runtime bundle. Second,
    default automatic reparameterization turned LogNormal Hill parameters into nested
    `*_base_decentered` sites, but the particle runtime reconstructed only direct `*_decentered`
    sites. The resolver now applies both the base location-scale reconstruction and the outer
    distribution transform. A deliberately tiny point-only 60-day execution smoke then completed
    the exact Euler--Maruyama particle path and posterior extraction for all nine latent states.

    A subsequent Modal A100-80GB run exercised the same restricted model through the complete
    production configuration: four chains, 4,000 warmup steps and 1,000 retained draws per chain,
    64 latent particles, two parameter particles, Pathfinder initialization and preconditioning,
    and the exact Euler--Maruyama marginalized Particle Gibbs kernel. The 459-dimensional parameter
    bundle compiled, all eight Pathfinder starts were finite, and inference produced 4,000 posterior
    draws plus latent paths of shape `(4, 1000, 10, 9)`. The inference transition took 693.6 seconds;
    inference plus posterior predictive checks took 760.6 seconds. The persisted run is
    `nof1-nine-construct-inference-results:/20260720T134757+0000`.

    This run also exposed a posterior-predictive runtime bug: posterior samples contain the raw
    ordered-logistic `obs_ordered_base` and `obs_ordered_gaps` sites, while the emission compiler
    consumes their derived `obs_ordered_cutpoints`. Prior prediction assembled that derived site,
    but posterior prediction did not. Posterior prediction now uses the canonical site registry to
    assemble derived likelihood parameters after draw subsampling. The repaired production PPC
    completed and emitted six overcoverage warnings, each reporting that all observations fell
    inside a 95% predictive interval and that the model may be too diffuse.

    Completion is not convergence. Maximum split \(\hat R\) was 2.87, minimum bulk ESS was 2.75,
    and minimum tail ESS was 4.51. Mean parameter acceptance was 0.366 and mean latent update
    fraction was 0.403, but several adapted latent proposal scales collapsed to the configured
    \(10^{-5}\) floor. These diagnostics, together with the PPC warnings, reject this fit as a
    basis for numeric or causal claims. They are evidence that production execution works for the
    restricted point-observation model and that the current sparse likelihood does not adequately
    identify its parameterization.

    Development reruns now use the canonical `scripts/cached_fit.py` Modal runner. Its JAX cache is
    persistent, and its content-addressed warmup artifact stores Pathfinder's selected Gaussian,
    exact chain initial positions, parameter preconditioner, and IEKS reference trajectories before
    particle MCMC begins. The cache fingerprint includes the complete compiled artifact, panel,
    source environment, and warmup policy, while deliberately excluding posterior sample budgets
    and particle count. This keeps iterative fits fast without reusing initialization after a
    structural, prior, data, code, seed, or warmup-policy change.

    Both inference runs are execution checks, not inferential results. The tiny local smoke used one
    warmup step, one draw, and three particles; the production-config run used the complete sampler
    budget, but both withheld the four interval-summary channels
    (`state_of_mind_valence`, `escitalopram_fill_quantity`,
    `missed_or_extra_escitalopram_dose_mentions`, and
    `external_stressor_event_count`). The unmodified nine-construct model still cannot enter the
    production particle sampler because interval-summary observations are rejected. Supporting
    those emissions in the exact particle likelihood is therefore the remaining blocker before a
    scientifically meaningful limited-model fit.
    """)
    return


@app.cell
def full_model_barrier(structural_plan, data_for_model, ps, workbench_run):
    barrier = (
        ps.validate_full_model(
            run=workbench_run,
            structural_plan=structural_plan,
            data_for_model=data_for_model,
        )
        if workbench_run.complete
        else None
    )
    return (barrier,)


@app.cell
def barrier_report(barrier, cs, mo):
    if barrier is None:
        _display = mo.md("## Full-model barrier\n\nWaiting for every construct to be admitted.")
    else:
        _shared = " · ".join(
            f"{timing.label}: {timing.duration_ms / 1000:.1f}s" for timing in barrier.timings
        )
        _reports = [cs.render_report(report.name, report) for report in barrier.reports]
        _display = mo.vstack([mo.md(f"## Full-model barrier\n\n{_shared}"), *_reports])
    mo.vstack([_display])
    return


if __name__ == "__main__":
    app.run()
