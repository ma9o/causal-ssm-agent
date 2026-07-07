"""Episode state machine: pure domain layer.

The pipeline is a state machine the navigating LLM traverses freely,
constrained only by artifact-level dependencies. This package holds the
engine-agnostic core:

- :mod:`artifacts` — artifact taxonomy, versions, provenance stamps
- :mod:`graph` — the artifact-level dependency DAG (stage specs)
- :mod:`hierarchy` — public actions, contexts, execution classes, write effects
- :mod:`moves` — ``legal_moves`` / ``apply_transition`` / staleness / freshness
- :mod:`errors` — typed stage-execution exceptions

Everything here is pure (no I/O, no engine imports) so it can run inside a
Temporal workflow sandbox, a test, or a notebook unchanged. I/O lives in
:mod:`nof1_causal_lab.machine.store` (versioned artifact store + journal) and
engine wiring lives in :mod:`nof1_causal_lab.machine.temporal`.
"""

from nof1_causal_lab.machine.artifacts import (
    ArtifactId,
    ArtifactVersionInfo,
    EpisodeState,
    Provenance,
)
from nof1_causal_lab.machine.errors import (
    ArtifactWriteRejected,
    ModelCompileError,
    ModelFitError,
    StageExecutionError,
)
from nof1_causal_lab.machine.graph import (
    ARTIFACT_GRAPH,
    ROOTS,
    WRITABLE_ARTIFACTS,
    Root,
    Transition,
    stage_spec,
)
from nof1_causal_lab.machine.hierarchy import (
    ACTIONS,
    CONTEXTS,
    ActionSpec,
    ContextSpec,
    action_spec,
    context_spec,
    legal_action_ids,
    legal_actions,
    primary_stage_action,
)
from nof1_causal_lab.machine.moves import (
    Move,
    RunStage,
    WriteArtifact,
    apply_transition,
    freshness_report,
    is_fresh,
    is_stale,
    legal_moves,
    validate_move,
)

__all__ = [
    "ARTIFACT_GRAPH",
    "ACTIONS",
    "ActionSpec",
    "ArtifactId",
    "ArtifactVersionInfo",
    "ArtifactWriteRejected",
    "CONTEXTS",
    "ContextSpec",
    "EpisodeState",
    "ModelCompileError",
    "ModelFitError",
    "Move",
    "Provenance",
    "ROOTS",
    "Root",
    "RunStage",
    "StageExecutionError",
    "Transition",
    "WRITABLE_ARTIFACTS",
    "WriteArtifact",
    "action_spec",
    "apply_transition",
    "context_spec",
    "freshness_report",
    "is_fresh",
    "is_stale",
    "legal_action_ids",
    "legal_actions",
    "legal_moves",
    "primary_stage_action",
    "stage_spec",
    "validate_move",
]
