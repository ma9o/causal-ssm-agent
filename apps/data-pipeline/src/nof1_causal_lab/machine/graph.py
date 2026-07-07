"""The artifact-level ruleset: how each artifact comes into existence.

Nodes are artifacts, not stages. An artifact enters the store exactly one of
three ways:

- **produced** by a run of a transition (named by its primary output; the
  ``stage_id`` is only the transition's execution/runner label),
- **written** directly (roots, and produced artifacts flagged ``writable``),
- **derived** — a deterministic milestone recomputed whenever its parent
  artifact is (re)created, by run *or* by write (``identification_report`` from
  ``causal_spec``). A derived artifact has no independent producer and is never
  written directly.

Run legality is a pure existence check over a transition's ``consumes`` — no
content predicates. The epistemic gate ("numeric claims only when
identification supports them") emerges structurally: ``causal_spec`` derives
``identification_report`` only when at least one treatment is explicitly
identifiable, and ``compiled_ssm`` consumes that milestone, so fitting and
interventions are simply never enabled without it.
"""

from __future__ import annotations

import graphlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId

# How a produced artifact is computed — and therefore whether an external agent
# can shortcut its run by writing the artifact itself (``judgment`` work is
# proposal work an agent can supply directly; the other two need compute/creds).
CreationClass = Literal["deterministic", "batch_llm", "judgment"]


@dataclass(frozen=True)
class Transition:
    """How a produced artifact is computed from its inputs.

    ``consumes`` are the inputs whose existence gates a run (the guard).
    ``produces`` is the primary artifact — the transition's identity.
    ``produces_optional`` are substantive co-outputs withheld on a negative
    finding (their absence is a finding, not a failure). ``derives`` are
    deterministic milestones recomputed on every creation of ``produces`` — by
    run *or* by a direct write of ``produces``. Re-running retracts an optional
    or derived artifact the new run withholds. ``writable`` says a caller may
    also supply ``produces`` directly via a ``write`` move.
    """

    stage_id: str
    consumes: tuple[ArtifactId, ...]
    produces: tuple[ArtifactId, ...]
    creation_class: CreationClass
    produces_optional: tuple[ArtifactId, ...] = ()
    derives: tuple[ArtifactId, ...] = ()
    writable: bool = False

    @property
    def all_produces(self) -> tuple[ArtifactId, ...]:
        return self.produces + self.produces_optional + self.derives


@dataclass(frozen=True)
class Root:
    """An artifact that enters the store by ``write``, not by a transition.

    ``write_pins`` are the inputs a direct write stamps into ``derived_from``,
    so a written artifact participates in staleness like a computed one. A pure
    root (``question``) pins nothing; ``saved_scenarios`` pins the ``posterior``
    it was simulated against.
    """

    artifact_id: ArtifactId
    write_pins: tuple[ArtifactId, ...] = ()


ARTIFACT_GRAPH: tuple[Transition, ...] = (
    Transition(
        stage_id="stage-0",
        consumes=(),
        produces=("raw_data",),
        creation_class="batch_llm",
    ),
    Transition(
        stage_id="stage-1a",
        consumes=("question",),
        produces=("constructs",),
        creation_class="judgment",
        writable=True,
    ),
    Transition(
        stage_id="stage-1b",
        consumes=("question", "raw_data", "constructs"),
        produces=("causal_spec",),
        creation_class="judgment",
        derives=("identification_report",),
        writable=True,
    ),
    Transition(
        stage_id="stage-2",
        consumes=("question", "raw_data", "causal_spec"),
        produces=("extraction_report",),
        creation_class="batch_llm",
        produces_optional=("model_data",),
        writable=True,
    ),
    Transition(
        stage_id="stage-3",
        consumes=("causal_spec", "model_data"),
        produces=("validation_report",),
        creation_class="deterministic",
        writable=True,
    ),
    Transition(
        stage_id="stage-4",
        consumes=(
            "question",
            "causal_spec",
            "identification_report",
            "model_data",
            "validation_report",
        ),
        produces=("compiled_ssm",),
        creation_class="judgment",
    ),
    Transition(
        stage_id="stage-5b",
        consumes=("compiled_ssm", "model_data"),
        produces=("posterior",),
        creation_class="deterministic",
    ),
    Transition(
        stage_id="stage-6",
        consumes=("posterior", "causal_spec", "identification_report"),
        produces=("baseline_ranking",),
        creation_class="judgment",
        writable=True,
    ),
)

# Roots enter the store via ``write`` moves, never via a transition.
ROOTS: tuple[Root, ...] = (
    Root(artifact_id="question"),
    Root(artifact_id="saved_scenarios", write_pins=("posterior",)),
)

ROOT_ARTIFACTS: tuple[ArtifactId, ...] = tuple(root.artifact_id for root in ROOTS)

# The full ``write`` surface: every root, plus every produced artifact whose
# transition is ``writable``. Derived milestones (``identification_report``) are
# deliberately absent — they are recomputed from their parent, never supplied.
WRITABLE_ARTIFACTS: tuple[ArtifactId, ...] = ROOT_ARTIFACTS + tuple(
    spec.produces[0] for spec in ARTIFACT_GRAPH if spec.writable
)


def stage_spec(stage_id: str) -> Transition:
    for spec in ARTIFACT_GRAPH:
        if spec.stage_id == stage_id:
            return spec
    known = ", ".join(spec.stage_id for spec in ARTIFACT_GRAPH)
    raise KeyError(f"Unknown stage '{stage_id}'. Expected one of: {known}")


def producer_of(artifact_id: ArtifactId) -> Transition | None:
    """The transition that produces or derives an artifact, or None for roots."""
    for spec in ARTIFACT_GRAPH:
        if artifact_id in spec.all_produces:
            return spec
    return None


def topological_stage_order() -> tuple[str, ...]:
    """Stages sorted by artifact dependencies (for default-policy drivers)."""
    producers: dict[ArtifactId, str] = {}
    for spec in ARTIFACT_GRAPH:
        for artifact in spec.all_produces:
            producers[artifact] = spec.stage_id
    dep_graph = {
        spec.stage_id: {producers[artifact] for artifact in spec.consumes if artifact in producers}
        for spec in ARTIFACT_GRAPH
    }
    return tuple(graphlib.TopologicalSorter(dep_graph).static_order())


def _assert_graph_consistent() -> None:
    produced: set[ArtifactId] = set()
    for spec in ARTIFACT_GRAPH:
        if len(spec.produces) != 1:
            raise AssertionError(f"{spec.stage_id} must have exactly one primary output")
        for artifact in spec.all_produces:
            if artifact in produced:
                raise AssertionError(f"Artifact '{artifact}' has two producers")
            produced.add(artifact)
    overlap = produced.intersection(ROOT_ARTIFACTS)
    if overlap:
        raise AssertionError(f"Root artifacts cannot have transition producers: {overlap}")
    for root in ROOTS:
        for pinned in root.write_pins:
            if pinned not in produced:
                raise AssertionError(f"Root '{root.artifact_id}' pins unknown artifact '{pinned}'")
    topological_stage_order()  # raises on cycles


_assert_graph_consistent()
