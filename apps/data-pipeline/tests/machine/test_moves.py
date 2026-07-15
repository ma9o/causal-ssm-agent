"""legal_moves / apply_transition / staleness / freshness semantics."""

from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
from nof1_causal_lab.machine.graph import WRITABLE_ARTIFACTS, transition_spec
from nof1_causal_lab.machine.moves import (
    RetractedArtifact,
    RunArtifact,
    WriteArtifact,
    apply_transition,
    freshness_report,
    input_pins,
    is_stale,
    legal_moves,
    run_retractions,
    validate_move,
)


def _version(artifact_id, version=1, derived_from=None, produced_by=None, provenance="computed"):
    return ArtifactVersionInfo(
        artifact_id=artifact_id,
        version=version,
        provenance=provenance,
        derived_from=derived_from or {},
        produced_by=produced_by,
        created_at="2026-07-03T00:00:00Z",
    )


def _state(*infos):
    return EpisodeState().with_versions(list(infos))


def _runnable(state):
    return {move.artifact_id for move in legal_moves(state) if isinstance(move, RunArtifact)}


class TestLegalMoves:
    def test_empty_state_enables_only_raw_data(self):
        assert _runnable(EpisodeState()) == {"raw_data"}

    def test_question_write_enables_latent_structure(self):
        state = _state(_version("question", provenance="human"))
        assert _runnable(state) == {"raw_data", "latent_structure"}

    def test_declared_writes_are_offered(self):
        offered = {
            move.artifact_id
            for move in legal_moves(EpisodeState())
            if isinstance(move, WriteArtifact)
        }
        assert offered == set(WRITABLE_ARTIFACTS)
        assert "question" in offered
        assert "measurement_structure" in offered
        assert "statistical_model_spec" in offered
        assert "causal_design" not in offered
        assert "raw_data" not in offered

    def test_no_identification_report_disables_fit_chain(self):
        state = _state(
            _version("question", provenance="human"),
            _version("raw_data"),
            _version("latent_structure"),
            _version("measurement_structure"),
            _version("causal_design"),
            _version("measurements"),
            _version("panel"),
            _version("validation_report"),
        )
        runnable = _runnable(state)
        assert "statistical_model_spec" not in runnable
        assert "measurements" in runnable

    def test_identification_report_enables_statistical_model_spec(self):
        state = _state(
            _version("question", provenance="human"),
            _version("raw_data"),
            _version("latent_structure"),
            _version("measurement_structure"),
            _version("causal_design"),
            _version("identification_report"),
            _version("measurements"),
            _version("panel"),
            _version("validation_report"),
        )
        assert "statistical_model_spec" in _runnable(state)

    def test_baseline_report_does_not_require_question(self):
        state = _state(
            _version("causal_design"),
            _version("identification_report"),
            _version("posterior"),
        )
        assert "baseline_report" in _runnable(state)


class TestValidateMove:
    def test_missing_inputs_rejected_with_names(self):
        reason = validate_move(EpisodeState(), RunArtifact(artifact_id="measurement_structure"))
        assert reason is not None
        assert "question" in reason
        assert "latent_structure" in reason

    def test_unknown_transition_rejected(self):
        reason = validate_move(EpisodeState(), RunArtifact(artifact_id="validation_report"))
        assert reason is not None
        assert "Unknown transition" in reason

    def test_computed_provenance_write_rejected(self):
        reason = validate_move(
            EpisodeState(),
            WriteArtifact(artifact_id="question", provenance="computed"),
        )
        assert reason is not None

    def test_derived_artifact_write_rejected(self):
        reason = validate_move(EpisodeState(), WriteArtifact(artifact_id="causal_design"))
        assert reason is not None
        assert "not writable" in reason

    def test_legal_run_accepted(self):
        assert validate_move(EpisodeState(), RunArtifact(artifact_id="raw_data")) is None


class TestApplyTransition:
    def test_produced_versions_become_current(self):
        state = apply_transition(EpisodeState(), [_version("raw_data")])
        assert state.has("raw_data")
        assert state.get("raw_data").version == 1

    def test_rerun_supersedes_version(self):
        state = _state(_version("raw_data", version=1))
        state = apply_transition(state, [_version("raw_data", version=2)])
        assert state.get("raw_data").version == 2

    def test_optional_artifact_retracted_when_withheld(self):
        spec = transition_spec("measurements")
        state = _state(
            _version("measurements", version=1),
            _version("panel", version=1),
        )
        produced = [_version("measurements", version=2)]
        retracted = run_retractions(state, spec, produced)
        assert retracted == [
            RetractedArtifact(
                artifact_id="panel",
                reason_ref="measurements.produces_optional.panel",
            )
        ]
        next_state = apply_transition(state, produced, retracted)
        assert not next_state.has("panel")
        assert next_state.get("measurements").version == 2

    def test_no_retraction_when_optional_still_produced(self):
        spec = transition_spec("measurements")
        state = _state(_version("panel", version=1))
        produced = [
            _version("measurements"),
            _version("panel", version=2),
        ]
        assert run_retractions(state, spec, produced) == []


class TestStaleness:
    def _fitted_chain(self):
        question = _version("question", provenance="human")
        raw = _version("raw_data", produced_by="run:raw_data")
        latent_structure = _version(
            "latent_structure", derived_from={"question": 1}, produced_by="run:latent_structure"
        )
        measurement_structure = _version(
            "measurement_structure",
            derived_from={"question": 1, "raw_data": 1, "latent_structure": 1},
            produced_by="run:measurement_structure",
        )
        causal_design = _version(
            "causal_design",
            derived_from={"latent_structure": 1, "measurement_structure": 1},
            produced_by="derive:causal_design",
        )
        identification_report = _version(
            "identification_report",
            derived_from={"causal_design": 1},
            produced_by="derive:identification_report",
        )
        measurements = _version(
            "measurements",
            derived_from={"question": 1, "raw_data": 1, "measurement_structure": 1},
            produced_by="run:measurements",
        )
        panel = _version(
            "panel",
            derived_from={"question": 1, "raw_data": 1, "measurement_structure": 1},
            produced_by="run:measurements",
        )
        validation = _version(
            "validation_report",
            derived_from={"panel": 1, "causal_design": 1},
            produced_by="derive:validation_report",
        )
        sms = _version(
            "statistical_model_spec",
            derived_from={
                "question": 1,
                "causal_design": 1,
                "identification_report": 1,
                "panel": 1,
                "validation_report": 1,
            },
            produced_by="run:statistical_model_spec",
        )
        compiled = _version(
            "compiled_ssm",
            derived_from={"statistical_model_spec": 1, "causal_design": 1},
            produced_by="derive:compiled_ssm",
        )
        posterior = _version(
            "posterior",
            derived_from={"compiled_ssm": 1, "panel": 1},
            produced_by="run:posterior",
        )
        return _state(
            question,
            raw,
            latent_structure,
            measurement_structure,
            causal_design,
            identification_report,
            measurements,
            panel,
            validation,
            sms,
            compiled,
            posterior,
        )

    def test_fresh_chain_is_not_stale(self):
        state = self._fitted_chain()
        assert state.has("posterior")
        assert not is_stale(state, "posterior")

    def test_editing_measurement_structure_stales_produced_descendants(self):
        state = self._fitted_chain()
        state = apply_transition(
            state,
            [
                _version("measurement_structure", version=2, provenance="human"),
                _version(
                    "causal_design",
                    version=2,
                    derived_from={"latent_structure": 1, "measurement_structure": 2},
                ),
                _version("identification_report", version=2, derived_from={"causal_design": 2}),
                _version(
                    "compiled_ssm",
                    version=2,
                    derived_from={"statistical_model_spec": 1, "causal_design": 2},
                ),
            ],
        )
        assert is_stale(state, "measurements")
        assert is_stale(state, "panel")
        assert is_stale(state, "statistical_model_spec")
        assert is_stale(state, "posterior")
        # Derived nodes are recomputed in the move and are never reported stale.
        assert not is_stale(state, "causal_design")
        assert not is_stale(state, "identification_report")
        assert not is_stale(state, "compiled_ssm")
        assert not is_stale(state, "question")
        assert not is_stale(state, "latent_structure")

    def test_retracted_input_stales_dependents(self):
        state = self._fitted_chain()
        state = state.without(["panel"])
        assert is_stale(state, "posterior")

    def test_absent_artifact_is_not_stale(self):
        assert not is_stale(EpisodeState(), "posterior")

    def test_recompute_restores_freshness(self):
        state = self._fitted_chain()
        state = apply_transition(
            state,
            [
                _version("measurement_structure", version=2),
                _version(
                    "causal_design",
                    version=2,
                    derived_from={"latent_structure": 1, "measurement_structure": 2},
                ),
                _version("identification_report", version=2, derived_from={"causal_design": 2}),
            ],
        )
        state = apply_transition(
            state,
            [
                _version(
                    "measurements",
                    version=2,
                    derived_from={"question": 1, "raw_data": 1, "measurement_structure": 2},
                ),
                _version(
                    "panel",
                    version=2,
                    derived_from={"question": 1, "raw_data": 1, "measurement_structure": 2},
                ),
                _version(
                    "validation_report",
                    version=2,
                    derived_from={"panel": 2, "causal_design": 2},
                ),
            ],
        )
        state = apply_transition(
            state,
            [
                _version(
                    "statistical_model_spec",
                    version=2,
                    derived_from={
                        "question": 1,
                        "causal_design": 2,
                        "identification_report": 2,
                        "panel": 2,
                        "validation_report": 2,
                    },
                )
            ],
        )
        state = apply_transition(
            state,
            [
                _version(
                    "compiled_ssm",
                    version=2,
                    derived_from={"statistical_model_spec": 2, "causal_design": 2},
                )
            ],
        )
        state = apply_transition(
            state,
            [
                _version(
                    "posterior",
                    version=2,
                    derived_from={"compiled_ssm": 2, "panel": 2},
                )
            ],
        )
        assert state.has("posterior")
        assert not is_stale(state, "posterior")


class TestInputPins:
    def test_pins_current_versions(self):
        state = _state(
            _version("question", version=3, provenance="human"),
        )
        pins = input_pins(state, transition_spec("latent_structure"))
        assert pins == {"question": 3}


def test_freshness_report_shape():
    state = _state(_version("question", provenance="human"))
    report = freshness_report(state)
    by_id = {status.artifact_id: status for status in report}
    assert by_id["question"].exists
    assert by_id["question"].provenance == "human"
    assert not by_id["posterior"].exists
    assert not by_id["posterior"].stale
