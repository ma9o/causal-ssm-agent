"""The artifact-level ruleset: how each artifact comes into existence.

Nodes are artifacts, not stages. An artifact enters the store exactly one of
three ways:

- **root** artifacts are written directly by a caller,
- **produced** artifacts are computed by running the transition keyed by the
  artifact it primarily creates,
- **derived** artifacts are deterministic machine-maintained nodes recomputed
  from their parents inside the same move that changed those parents.

Run legality is a pure existence check over a transition's ``consumes``. Derived
nodes are never runnable, never writable, and never stale: they move in lockstep
with their parents through the derivation cascade.
"""

from __future__ import annotations

import graphlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nof1_causal_lab.machine.artifacts import ArtifactId

# How a produced artifact is computed, and therefore whether an external agent
# can shortcut its run by writing the artifact itself.
CreationClass = Literal["deterministic", "batch_llm", "judgment"]


@dataclass(frozen=True)
class Transition:
    """How a produced artifact is computed from its inputs.

    ``produces`` is the transition identity.
    """

    consumes: tuple[ArtifactId, ...]
    produces: ArtifactId
    creation_class: CreationClass
    produces_optional: tuple[ArtifactId, ...] = ()
    writable: bool = False

    @property
    def transition_id(self) -> ArtifactId:
        return self.produces

    @property
    def all_produces(self) -> tuple[ArtifactId, ...]:
        return (self.produces, *self.produces_optional)


@dataclass(frozen=True)
class Derivation:
    """A deterministic, machine-maintained artifact node."""

    produces: ArtifactId
    from_: tuple[ArtifactId, ...]
    optional: bool = False


@dataclass(frozen=True)
class Root:
    """An artifact that enters the store by ``write``, not by a transition."""

    artifact_id: ArtifactId
    write_pins: tuple[ArtifactId, ...] = ()


ARTIFACT_GRAPH: tuple[Transition, ...] = (
    Transition(
        consumes=(),
        produces="raw_data",
        creation_class="batch_llm",
    ),
    Transition(
        consumes=("question",),
        produces="latent_structure",
        creation_class="judgment",
        writable=True,
    ),
    Transition(
        consumes=("question", "raw_data", "latent_structure"),
        produces="measurement_structure",
        creation_class="judgment",
        writable=True,
    ),
    Transition(
        consumes=("question", "raw_data", "measurement_structure"),
        produces="measurements",
        creation_class="batch_llm",
        produces_optional=("panel",),
    ),
    Transition(
        consumes=(
            "question",
            "causal_design",
            "identification_report",
            "panel",
            "validation_report",
        ),
        produces="statistical_model_spec",
        creation_class="judgment",
        writable=True,
    ),
    Transition(
        consumes=("compiled_ssm", "panel"),
        produces="posterior",
        creation_class="deterministic",
    ),
    Transition(
        consumes=("posterior", "causal_design", "identification_report"),
        produces="baseline_report",
        creation_class="judgment",
        writable=True,
    ),
)

DERIVATIONS: tuple[Derivation, ...] = (
    Derivation(produces="causal_design", from_=("latent_structure", "measurement_structure")),
    Derivation(produces="identification_report", from_=("causal_design",), optional=True),
    Derivation(produces="validation_report", from_=("panel", "causal_design")),
    Derivation(produces="compiled_ssm", from_=("statistical_model_spec", "causal_design")),
)

ROOTS: tuple[Root, ...] = (
    Root(artifact_id="question"),
    Root(artifact_id="saved_scenarios", write_pins=("posterior",)),
)

ROOT_ARTIFACTS: tuple[ArtifactId, ...] = tuple(root.artifact_id for root in ROOTS)

# The full ``write`` surface: every root, plus every judgment-class transition.
WRITABLE_ARTIFACTS: tuple[ArtifactId, ...] = ROOT_ARTIFACTS + tuple(
    spec.transition_id for spec in ARTIFACT_GRAPH if spec.writable
)


def transition_spec(artifact_id: ArtifactId) -> Transition:
    """Return the transition whose primary output is ``artifact_id``."""
    for spec in ARTIFACT_GRAPH:
        if spec.transition_id == artifact_id:
            return spec
    known = ", ".join(spec.transition_id for spec in ARTIFACT_GRAPH)
    raise KeyError(f"Unknown transition '{artifact_id}'. Expected one of: {known}")


def producer_of(artifact_id: ArtifactId) -> Transition | Derivation | None:
    """The graph node that creates an artifact, or None for roots."""
    for spec in ARTIFACT_GRAPH:
        if artifact_id in spec.all_produces:
            return spec
    for spec in DERIVATIONS:
        if artifact_id == spec.produces:
            return spec
    return None


def topological_derivation_order() -> tuple[Derivation, ...]:
    """Derived nodes sorted so parent derivations precede their children."""
    derived_by_id = {spec.produces: spec for spec in DERIVATIONS}
    dep_graph = {
        spec.produces: {parent for parent in spec.from_ if parent in derived_by_id}
        for spec in DERIVATIONS
    }
    ordered = graphlib.TopologicalSorter(dep_graph).static_order()
    return tuple(derived_by_id[artifact_id] for artifact_id in ordered)


def _transition_dependencies(artifact_id: ArtifactId) -> set[ArtifactId]:
    """Runnable transition ids upstream of an artifact dependency."""
    producer = producer_of(artifact_id)
    if producer is None:
        return set()
    if isinstance(producer, Transition):
        return {producer.transition_id}
    dependencies: set[ArtifactId] = set()
    for parent in producer.from_:
        dependencies.update(_transition_dependencies(parent))
    return dependencies


def topological_transition_order() -> tuple[ArtifactId, ...]:
    """Runnable transitions sorted by artifact dependencies."""
    dep_graph = {
        spec.transition_id: {
            dependency
            for artifact in spec.consumes
            for dependency in _transition_dependencies(artifact)
            if dependency != spec.transition_id
        }
        for spec in ARTIFACT_GRAPH
    }
    return tuple(graphlib.TopologicalSorter(dep_graph).static_order())


def topological_artifact_order() -> tuple[ArtifactId, ...]:
    """Artifacts sorted by the complete artifact dependency graph."""
    from nof1_causal_lab.machine.artifacts import ARTIFACT_IDS

    dependencies = _artifact_dependency_graph()
    for artifact_id in ARTIFACT_IDS:
        dependencies.setdefault(artifact_id, set())
    return tuple(graphlib.TopologicalSorter(dependencies).static_order())


def _artifact_dependency_graph() -> dict[ArtifactId, set[ArtifactId]]:
    dependencies: dict[ArtifactId, set[ArtifactId]] = {}
    for spec in ARTIFACT_GRAPH:
        for artifact in spec.all_produces:
            dependencies[artifact] = set(spec.consumes)
    for spec in DERIVATIONS:
        dependencies[spec.produces] = set(spec.from_)
    return dependencies


def _assert_graph_consistent() -> None:
    from nof1_causal_lab.machine.artifacts import ARTIFACT_IDS

    produced: set[ArtifactId] = set()
    for spec in ARTIFACT_GRAPH:
        if spec.writable and spec.creation_class != "judgment":
            raise AssertionError(f"Writable transition '{spec.transition_id}' must be judgment")
        for artifact in spec.all_produces:
            if artifact in produced:
                raise AssertionError(f"Artifact '{artifact}' has two producers")
            produced.add(artifact)

    for spec in DERIVATIONS:
        if spec.produces in produced:
            raise AssertionError(f"Derived artifact '{spec.produces}' has a transition producer")
        produced.add(spec.produces)

    overlap = produced.intersection(ROOT_ARTIFACTS)
    if overlap:
        raise AssertionError(f"Root artifacts cannot have producers: {overlap}")

    all_artifacts = set(ARTIFACT_IDS)
    unknown_produced = produced - all_artifacts
    if unknown_produced:
        raise AssertionError(f"Graph produces unknown artifacts: {unknown_produced}")

    missing = all_artifacts - produced - set(ROOT_ARTIFACTS)
    if missing:
        raise AssertionError(f"Artifacts need a root, transition, or derivation: {missing}")

    dependencies = _artifact_dependency_graph()
    for artifact, parents in dependencies.items():
        unknown_parents = parents - all_artifacts
        if unknown_parents:
            raise AssertionError(f"{artifact} depends on unknown artifacts: {unknown_parents}")

    for root in ROOTS:
        unknown_pins = set(root.write_pins) - all_artifacts
        if unknown_pins:
            raise AssertionError(
                f"Root '{root.artifact_id}' pins unknown artifacts: {unknown_pins}"
            )

    graphlib.TopologicalSorter(dependencies).static_order()
    topological_artifact_order()
    topological_transition_order()
    topological_derivation_order()


_assert_graph_consistent()
