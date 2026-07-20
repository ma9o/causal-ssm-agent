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

    return Path, build_causal_design, build_construct_units, cs, json, pl, ps


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    # Codex-driven production prior specification

    This is an **authoring workbench**, not another embedded agent. The `PROPOSALS` cell below
    is the judgment surface: Codex edits ordinary Python payloads, reads the resulting
    diagnostics, and revises them directly.

    Everything after input loading is production code. Each payload goes through
    `ConstructBuildState.submit_construct`, so parameter membership, likelihood compatibility,
    compiler binding, exact nonlinear Diffrax prior prediction, and the C1--C5 reachability
    battery are the same ones used by Stage 4. Once all constructs are admitted, the notebook
    runs the same shared-draw full-model barrier. It deliberately does **not** call Temporal,
    `ArtifactStore`, checkpoints, telemetry, or the Pi harness.

    The current DEMO artifact files are read directly only to snapshot the four Stage-4 inputs
    into memory while storage is being refactored. The stored causal-design snapshot predates the
    known-input authoring repair, so this trial explicitly re-derives it in memory through the
    production causal-design assembler with the taper regime declared as a known transition input.
    The stored artifacts are not mutated. If their physical location changes, only the input cell
    changes; the authoring and validation path does not.
    """)
    return


@app.cell
def input_paths(Path):
    WORKSPACE_STORE = Path(__file__).resolve().parents[3] / "data/DEMO/store"
    QUESTION_PATH = WORKSPACE_STORE / "question/v1/question.json"
    CAUSAL_DESIGN_PATH = WORKSPACE_STORE / "causal_design/v1/causal_design.json"
    PANEL_PATH = WORKSPACE_STORE / "panel/v1/panel.parquet"
    VALIDATION_REPORT_PATH = WORKSPACE_STORE / "validation_report/v1/validation_report.json"
    return CAUSAL_DESIGN_PATH, PANEL_PATH, QUESTION_PATH, VALIDATION_REPORT_PATH, WORKSPACE_STORE


@app.cell
def load_input_snapshot(
    CAUSAL_DESIGN_PATH,
    PANEL_PATH,
    QUESTION_PATH,
    VALIDATION_REPORT_PATH,
    build_causal_design,
    json,
    pl,
):
    question = json.loads(QUESTION_PATH.read_text())["text"]
    _stored_causal_design = json.loads(CAUSAL_DESIGN_PATH.read_text())["causal_design"]
    _known_inputs = [
        {
            "construct": "counterfactual_taper_regime",
            "source_indicator": "observed_taper_regime_indicator",
            "scale": 1.0,
            "missing_policy": "forward_fill",
        }
    ]
    causal_design = build_causal_design(
        _stored_causal_design["latent"],
        _stored_causal_design["measurement"],
        _stored_causal_design.get("identifiability"),
        known_inputs=_known_inputs,
    )
    data_for_model = pl.read_parquet(PANEL_PATH)
    validation_report = json.loads(VALIDATION_REPORT_PATH.read_text())
    indicator_audits = validation_report["indicators"]
    return causal_design, data_for_model, indicator_audits, question, validation_report


@app.cell(hide_code=True)
def input_audit(build_construct_units, causal_design, data_for_model, mo, validation_report):
    _units = build_construct_units(causal_design)
    _construct_count = len(causal_design["latent"]["constructs"])
    _indicator_count = len(causal_design["measurement"]["indicators"])
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
        | known transition inputs | {len(causal_design["estimation"]["known_inputs"])} |
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
    - Accept a soft check only with its exact `(check, target)` pair and a substantive rationale.
      Hard checks are never overridden.
    - Do not author compiler defaults or parameters absent from the production prompt shown below.
    - Parameters marked conditional in the production prompt are authorable surfaces, not mandatory
      priors: include them only when the submitted family and link activate them.
    - For a time-invariant construct without a standardized channel, omit `t0_mean_*`: the
      compiler fixes the latent location at zero and leaves the channel-side location parameter
      free. A free latent mean and a free channel location would violate the location-anchor
      invariant.
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


@app.cell
def authored_proposals():
    N_DRAWS = 16
    SEED = 20260716
    PROPOSALS = [
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
    return N_DRAWS, PROPOSALS, SEED


@app.cell
def replay_proposals(N_DRAWS, PROPOSALS, SEED, causal_design, data_for_model, ps):
    workbench_run = ps.run_authored_proposals(
        causal_design=causal_design,
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
        _title = f"{_attempt.construct} — authored attempt {_attempt.attempt}"
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
    causal_design,
    mo,
    ps,
    question,
    validation_report,
    workbench_run,
):
    _prompt = ps.next_construct_prompt(
        run=workbench_run,
        question=question,
        causal_design=causal_design,
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
def next_indicator_audits(causal_design, indicator_audits, json, mo, workbench_run):
    _construct = workbench_run.state.current_construct
    _names = [
        indicator["name"]
        for indicator in causal_design["measurement"]["indicators"]
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
    follow the same location-anchor rule. The prompt now exposes active likelihood-extra priors such
    as `obs_ordered_base` and `obs_ordered_gaps`, and the compiler independently rejects an authored
    intercept on an inactive surface.

    The stored DEMO snapshot's empty `known_inputs` declaration remains a documented input
    prerequisite; this trial re-derives the executable design through the production assembler so
    the repaired known-input role is exercised.
    """)
    return


@app.cell
def full_model_barrier(causal_design, data_for_model, ps, workbench_run):
    barrier = (
        ps.validate_full_model(
            run=workbench_run,
            causal_design=causal_design,
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
