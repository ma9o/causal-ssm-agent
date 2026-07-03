"""The artifact-level dependency DAG.

Nodes are artifacts, not stages; a stage is a transition that consumes
artifact versions and produces artifact versions. Enabledness is a pure
existence check over consumed artifacts — no content predicates. The
epistemic gate ("numeric claims only when identification supports them")
emerges structurally: stage-1b produces ``estimands`` only when nonempty,
so fitting and interventions are simply never enabled without it.
"""

from __future__ import annotations

import graphlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId


@dataclass(frozen=True)
class StageSpec:
    """Declarative transition metadata: what a stage consumes and produces.

    ``produces`` artifacts always appear on success; ``produces_optional``
    artifacts appear only when the stage's finding is nonempty (their absence
    is a negative finding, not a failure). Re-running the stage retracts a
    previously produced optional artifact if the new run withholds it.
    """

    stage_id: str
    consumes: tuple[ArtifactId, ...]
    produces: tuple[ArtifactId, ...]
    produces_optional: tuple[ArtifactId, ...] = ()

    @property
    def all_produces(self) -> tuple[ArtifactId, ...]:
        return self.produces + self.produces_optional


# The raw ``question`` text is consumed by stage-2 and stage-4 today; both
# edges are slated to be cut (extraction should be driven by the measurement
# model's indicators, elicitation by the structured spec — the latter lands
# with the stage-4 rework). Keep the graph honest until the runners change.
ARTIFACT_GRAPH: tuple[StageSpec, ...] = (
    StageSpec(
        stage_id="stage-0",
        consumes=(),
        produces=("raw_data",),
    ),
    StageSpec(
        stage_id="stage-1a",
        consumes=("question",),
        produces=("constructs",),
    ),
    StageSpec(
        stage_id="stage-1b",
        consumes=("question", "raw_data", "constructs"),
        produces=("causal_spec", "identification_report"),
        produces_optional=("estimands",),
    ),
    StageSpec(
        stage_id="stage-2",
        consumes=("question", "raw_data", "causal_spec"),
        produces=("extraction_report",),
        produces_optional=("model_data",),
    ),
    StageSpec(
        stage_id="stage-3",
        consumes=("causal_spec", "model_data"),
        produces=("validation_report",),
    ),
    StageSpec(
        stage_id="stage-4",
        consumes=("question", "causal_spec", "estimands", "model_data", "validation_report"),
        produces=("compiled_ssm",),
    ),
    StageSpec(
        stage_id="stage-5b",
        consumes=("compiled_ssm", "model_data"),
        produces=("posterior",),
    ),
    StageSpec(
        stage_id="stage-6",
        consumes=("posterior", "causal_spec", "estimands"),
        produces=("baseline_ranking",),
    ),
)

# Root artifacts enter the store via ``write`` moves, never via stages.
ROOT_ARTIFACTS: tuple[ArtifactId, ...] = ("question", "saved_scenarios")


def stage_spec(stage_id: str) -> StageSpec:
    for spec in ARTIFACT_GRAPH:
        if spec.stage_id == stage_id:
            return spec
    known = ", ".join(spec.stage_id for spec in ARTIFACT_GRAPH)
    raise KeyError(f"Unknown stage '{stage_id}'. Expected one of: {known}")


def producer_of(artifact_id: ArtifactId) -> StageSpec | None:
    """The stage that produces an artifact, or None for root artifacts."""
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
        for artifact in spec.all_produces:
            if artifact in produced:
                raise AssertionError(f"Artifact '{artifact}' has two producers")
            produced.add(artifact)
    overlap = produced.intersection(ROOT_ARTIFACTS)
    if overlap:
        raise AssertionError(f"Root artifacts cannot have stage producers: {overlap}")
    topological_stage_order()  # raises on cycles


_assert_graph_consistent()
