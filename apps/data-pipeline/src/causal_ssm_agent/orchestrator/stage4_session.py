"""Stage 4 session ownership and turn tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .stage4_navigation import get_active_prompt_block
from .stage4_reducer import _compute_stage4_validate_step_with_transitions
from .stage4_state import Stage4AcceptedState, Stage4DoneCursor, Stage4Runtime
from .stage4_submission import get_stage4_block_handler
from .stage4_types import Stage4Deps, Stage4Result

if TYPE_CHECKING:
    from collections.abc import Callable

    from .stage4_orchestrator import Stage4FrontierBlock, Stage4Plan
    from .stage4_prompt_context import Stage4Messages, Stage4Turn


@dataclass(frozen=True)
class Stage4TurnOutcome:
    """Structured outcome for one Stage 4 model turn."""

    block_id: str
    validate_submitted: bool
    submit_count: int
    latest_feedback: str | None
    next_block_id: str | None


@dataclass
class _Stage4TurnTracker:
    """Mutable tracker for explicit tool submissions inside one model turn."""

    block_id: str
    submit_count: int = 0
    latest_feedback: str | None = None
    next_block_id: str | None = None


@dataclass
class Stage4Session:
    """Single owner of the current Stage 4 turn and accepted state."""

    plan: Stage4Plan
    prompt_context: Stage4Messages
    deps: Stage4Deps
    runtime: Stage4Runtime = field(default_factory=Stage4Runtime)
    persist_runtime: Callable[[Stage4Runtime, tuple[dict[str, Any], ...]], None] | None = None
    _turn_tracker: _Stage4TurnTracker | None = field(default=None, init=False, repr=False)

    @property
    def accepted(self) -> Stage4AcceptedState:
        return self.runtime.accepted

    @property
    def search_cache(self) -> dict[str, str]:
        return self.runtime.search_cache

    @property
    def search_queries(self) -> dict[str, str]:
        return self.runtime.search_queries

    def current_block(self) -> Stage4FrontierBlock | None:
        """Return the active reducer block, if any."""
        return get_active_prompt_block(self.plan, self.runtime)

    def current_turn(self) -> Stage4Turn | None:
        """Return the active prompt/tool turn, if any."""
        block = self.current_block()
        if block is None:
            return None
        handler = get_stage4_block_handler(block.kind)
        return self.prompt_context.render_turn(
            plan=self.plan,
            runtime=self.runtime,
            block=block,
            handler=handler,
        )

    def begin_turn(self, block_id: str) -> None:
        """Start tracking explicit submissions for one model turn."""
        if self._turn_tracker is not None:
            raise ValueError(
                f"Stage 4 turn tracking already active for block {self._turn_tracker.block_id!r}"
            )
        self._turn_tracker = _Stage4TurnTracker(block_id=block_id)

    def finish_turn(self, block_id: str) -> Stage4TurnOutcome:
        """Finish the active model turn and return its explicit submission outcome."""
        tracker = self._turn_tracker
        if tracker is None:
            raise ValueError("Stage 4 turn tracking was not started before finish_turn()")
        if tracker.block_id != block_id:
            raise ValueError(
                f"Stage 4 turn tracking mismatch: expected {tracker.block_id!r}, got {block_id!r}"
            )
        self._turn_tracker = None
        return Stage4TurnOutcome(
            block_id=tracker.block_id,
            validate_submitted=tracker.submit_count > 0,
            submit_count=tracker.submit_count,
            latest_feedback=tracker.latest_feedback,
            next_block_id=tracker.next_block_id,
        )

    def discard_turn(self) -> None:
        """Clear any active turn tracker after an aborted model call."""
        self._turn_tracker = None

    def submit(self, payload: dict[str, Any]) -> str:
        """Apply one block-local submission and return reducer feedback."""
        _stage_output, feedback, transitions = _compute_stage4_validate_step_with_transitions(
            payload,
            plan=self.plan,
            runtime=self.runtime,
            deps=self.deps,
        )
        if self.persist_runtime is not None:
            self.persist_runtime(self.runtime, transitions)
        if self._turn_tracker is not None:
            next_block = self.current_block()
            self._turn_tracker.submit_count += 1
            self._turn_tracker.latest_feedback = feedback
            self._turn_tracker.next_block_id = None if next_block is None else next_block.id
        return feedback

    def is_done(self) -> bool:
        """Whether Stage 4 has produced a final accepted result."""
        return (
            isinstance(self.runtime.cursor, Stage4DoneCursor)
            and self.accepted.model_spec is not None
            and bool(self.accepted.authored_priors)
        )

    def result(self) -> Stage4Result:
        """Materialize the current accepted Stage 4 result."""
        if self.accepted.model_spec is None or not self.accepted.authored_priors:
            raise ValueError("Stage 4 session has not completed a valid model_spec + priors")
        return Stage4Result(
            model_spec=self.accepted.model_spec,
            authored_priors=self.accepted.authored_priors,
            search_queries=dict(self.search_queries),
            validation=self.accepted.validation,
        )
