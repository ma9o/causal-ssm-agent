"""Typed transition exceptions.

Discriminator: an exception means the transition FAILED TO EXECUTE ITS
CONTRACT — state must not change, and the failure is a property of the
attempt (recorded in the transition log), never of the world. Negative
findings (no estimable treatments, zero observations extracted) are NOT
exceptions: those transitions succeed and simply withhold their enabling
artifact while producing a report.

Temporal maps these to non-retryable ApplicationErrors; transient infra
failures (network, OOM, Modal preemption) stay as ordinary exceptions and
are retried by policy without the navigator ever seeing them.
"""

from __future__ import annotations

from nof1_causal_lab.json_types import UncheckedJsonObject  # noqa: TC001


class TransitionExecutionError(Exception):
    """A transition failed to produce its output artifacts.

    ``diagnostics`` carries whatever partial information the attempt yielded
    (e.g. sampler diagnostics from a diverged fit) — informative for the
    navigator, but never persisted as a poisoned pseudo-artifact.
    """

    def __init__(
        self,
        message: str,
        *,
        transition_id: str,
        diagnostics: UncheckedJsonObject | None = None,
    ) -> None:
        super().__init__(message)
        self.transition_id = transition_id
        self.diagnostics = diagnostics or {}


class ModelCompileError(TransitionExecutionError):
    """The model-spec transition could not compile an executable SSM."""


class ModelFitError(TransitionExecutionError):
    """The posterior transition failed to produce a usable fit."""


class ArtifactWriteRejected(ValueError):
    """A ``write`` move's payload failed schema validation."""

    def __init__(self, message: str, *, artifact_id: str) -> None:
        super().__init__(message)
        self.artifact_id = artifact_id
