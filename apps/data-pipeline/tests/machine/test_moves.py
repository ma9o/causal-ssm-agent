"""legal_moves / apply_transition / staleness / freshness semantics."""

from nof1_causal_lab.machine.artifacts import ArtifactVersionInfo, EpisodeState
from nof1_causal_lab.machine.graph import WRITABLE_ARTIFACTS, stage_spec
from nof1_causal_lab.machine.moves import (
    RunStage,
    WriteArtifact,
    apply_transition,
    freshness_report,
    input_pins,
    is_fresh,
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
    return {move.stage_id for move in legal_moves(state) if isinstance(move, RunStage)}


class TestLegalMoves:
    def test_empty_state_enables_only_stage0(self):
        assert _runnable(EpisodeState()) == {"stage-0"}

    def test_question_write_enables_stage1a(self):
        state = _state(_version("question", provenance="human"))
        assert _runnable(state) == {"stage-0", "stage-1a"}

    def test_declared_writes_are_offered(self):
        offered = {
            move.artifact_id
            for move in legal_moves(EpisodeState())
            if isinstance(move, WriteArtifact)
        }
        assert offered == set(WRITABLE_ARTIFACTS)
        assert "question" in offered
        assert "causal_spec" in offered
        assert "raw_data" not in offered

    def test_no_identification_report_disables_fit_chain(self):
        """The epistemic gate: identification produced nothing estimable."""
        state = _state(
            _version("question", provenance="human"),
            _version("raw_data", produced_by="stage-0"),
            _version("constructs", produced_by="stage-1a"),
            _version("causal_spec", produced_by="stage-1b"),
            _version("extraction_report", produced_by="stage-2"),
            _version("model_data", produced_by="stage-2"),
            _version("validation_report", produced_by="stage-3"),
        )
        runnable = _runnable(state)
        assert "stage-4" not in runnable
        assert "stage-3" in runnable

    def test_identification_report_enables_stage4(self):
        state = _state(
            _version("question", provenance="human"),
            _version("raw_data"),
            _version("constructs"),
            _version("causal_spec"),
            _version("identification_report"),
            _version("extraction_report"),
            _version("model_data"),
            _version("validation_report"),
        )
        assert "stage-4" in _runnable(state)

    def test_stage6_does_not_require_question(self):
        state = _state(
            _version("causal_spec"),
            _version("identification_report"),
            _version("posterior"),
        )
        assert "stage-6" in _runnable(state)


class TestValidateMove:
    def test_missing_inputs_rejected_with_names(self):
        reason = validate_move(EpisodeState(), RunStage(stage_id="stage-1b"))
        assert reason is not None
        assert "question" in reason
        assert "constructs" in reason

    def test_unknown_stage_rejected(self):
        reason = validate_move(EpisodeState(), RunStage(stage_id="stage-99"))
        assert reason is not None
        assert "Unknown stage" in reason

    def test_computed_provenance_write_rejected(self):
        reason = validate_move(
            EpisodeState(),
            WriteArtifact(artifact_id="question", provenance="computed"),
        )
        assert reason is not None

    def test_legal_run_accepted(self):
        assert validate_move(EpisodeState(), RunStage(stage_id="stage-0")) is None


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
        """A rerun of stage-1b that finds nothing estimable retracts the report."""
        spec = stage_spec("stage-1b")
        state = _state(
            _version("causal_spec", version=1),
            _version("identification_report", version=1),
        )
        produced = [
            _version("causal_spec", version=2),
        ]
        retracted = run_retractions(state, spec, produced)
        assert retracted == ["identification_report"]
        next_state = apply_transition(state, produced, retracted)
        assert not next_state.has("identification_report")
        assert next_state.get("causal_spec").version == 2

    def test_no_retraction_when_optional_still_produced(self):
        spec = stage_spec("stage-1b")
        state = _state(_version("identification_report", version=1))
        produced = [
            _version("causal_spec"),
            _version("identification_report", version=2),
        ]
        assert run_retractions(state, spec, produced) == []


class TestStaleness:
    def _fitted_chain(self):
        question = _version("question", provenance="human")
        raw = _version("raw_data", produced_by="stage-0")
        constructs = _version("constructs", derived_from={"question": 1}, produced_by="stage-1a")
        spec = _version(
            "causal_spec",
            derived_from={"question": 1, "raw_data": 1, "constructs": 1},
            produced_by="stage-1b",
        )
        model_data = _version(
            "model_data",
            derived_from={"question": 1, "raw_data": 1, "causal_spec": 1},
            produced_by="stage-2",
        )
        posterior = _version(
            "posterior",
            derived_from={"compiled_ssm": 1, "model_data": 1},
            produced_by="stage-5b",
        )
        compiled = _version(
            "compiled_ssm",
            derived_from={"causal_spec": 1, "model_data": 1},
            produced_by="stage-4",
        )
        return _state(question, raw, constructs, spec, model_data, compiled, posterior)

    def test_fresh_chain_reports_fresh(self):
        state = self._fitted_chain()
        assert is_fresh(state, "posterior")
        assert not is_stale(state, "posterior")

    def test_editing_causal_spec_stales_posterior_transitively(self):
        """The scenario halt-on-fail used to (badly) protect against."""
        state = self._fitted_chain()
        state = apply_transition(
            state,
            [_version("causal_spec", version=2, provenance="human")],
        )
        assert is_stale(state, "model_data")
        assert is_stale(state, "compiled_ssm")
        assert is_stale(state, "posterior")
        assert not is_fresh(state, "posterior")
        # Roots and untouched intermediates stay fresh.
        assert not is_stale(state, "question")
        assert not is_stale(state, "constructs")

    def test_retracted_input_stales_dependents(self):
        state = self._fitted_chain()
        state = state.without(["model_data"])
        assert is_stale(state, "posterior")

    def test_absent_artifact_is_not_stale(self):
        assert not is_stale(EpisodeState(), "posterior")

    def test_recompute_restores_freshness(self):
        state = self._fitted_chain()
        state = apply_transition(state, [_version("causal_spec", version=2)])
        state = apply_transition(
            state,
            [
                _version(
                    "model_data",
                    version=2,
                    derived_from={"question": 1, "raw_data": 1, "causal_spec": 2},
                )
            ],
        )
        state = apply_transition(
            state,
            [
                _version(
                    "compiled_ssm",
                    version=2,
                    derived_from={"causal_spec": 2, "model_data": 2},
                )
            ],
        )
        state = apply_transition(
            state,
            [
                _version(
                    "posterior",
                    version=2,
                    derived_from={"compiled_ssm": 2, "model_data": 2},
                )
            ],
        )
        assert is_fresh(state, "posterior")


class TestInputPins:
    def test_pins_current_versions(self):
        state = _state(
            _version("question", version=3, provenance="human"),
        )
        pins = input_pins(state, stage_spec("stage-1a"))
        assert pins == {"question": 3}


def test_freshness_report_shape():
    state = _state(_version("question", provenance="human"))
    report = freshness_report(state)
    by_id = {status.artifact_id: status for status in report}
    assert by_id["question"].exists
    assert by_id["question"].provenance == "human"
    assert not by_id["posterior"].exists
    assert not by_id["posterior"].stale
