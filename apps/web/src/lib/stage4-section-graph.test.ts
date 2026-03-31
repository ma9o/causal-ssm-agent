import {
  deriveStage4SectionEdges,
  getStage4SectionRect,
  routeStage4SectionEdge,
  type Stage4Point,
  type Stage4SectionId,
} from "./stage4-section-graph";
import { describe, expect, it } from "vitest";

function segmentIntersectsRect(
  start: Stage4Point,
  end: Stage4Point,
  rect: ReturnType<typeof getStage4SectionRect>,
): boolean {
  const minX = Math.min(start.x, end.x);
  const maxX = Math.max(start.x, end.x);
  const minY = Math.min(start.y, end.y);
  const maxY = Math.max(start.y, end.y);

  if (start.x === end.x) {
    const x = start.x;
    const overlapsX = x > rect.left && x < rect.right;
    const overlapsY = maxY > rect.top && minY < rect.bottom;
    return overlapsX && overlapsY;
  }

  if (start.y === end.y) {
    const y = start.y;
    const overlapsY = y > rect.top && y < rect.bottom;
    const overlapsX = maxX > rect.left && minX < rect.right;
    return overlapsY && overlapsX;
  }

  return false;
}

describe("stage4-section-graph", () => {
  it("collapses backend edges into direct category transitions", () => {
    const graph = {
      nodes: [
        { id: "indicator:x", kind: "indicator_decision", label: "X", phase: "model_decisions" },
        { id: "__lock__", kind: "model_spec_lock", label: "Lock", phase: "model_decisions" },
        { id: "review:model_spec", kind: "global_review", label: "Review", phase: "global_review" },
        { id: "effects:y", kind: "effect_prior", label: "Effects", phase: "prior_blocks" },
        { id: "__repair_barrier__", kind: "repair_barrier", label: "Repair", phase: "prior_blocks" },
        {
          id: "review:prior_system",
          kind: "global_prior_review",
          label: "Prior review",
          phase: "global_prior_review",
        },
        { id: "__done__", kind: "done", label: "Done", phase: "done" },
      ],
      edges: [
        { from: "indicator:x", to: "__lock__", kind: "phase_advance" },
        { from: "__lock__", to: "review:model_spec", kind: "phase_advance" },
        { from: "review:model_spec", to: "effects:y", kind: "phase_advance" },
        { from: "effects:y", to: "__repair_barrier__", kind: "repair_transition" },
        { from: "effects:y", to: "review:prior_system", kind: "repair_transition" },
        { from: "__repair_barrier__", to: "review:prior_system", kind: "repair_transition" },
        { from: "__repair_barrier__", to: "__done__", kind: "repair_transition" },
        { from: "effects:y", to: "__done__", kind: "phase_advance" },
        { from: "review:prior_system", to: "__done__", kind: "phase_advance" },
      ],
      phases: [],
    };

    expect(deriveStage4SectionEdges(graph)).toEqual([
      { from: "model_decisions", to: "global_review", kind: "phase_advance" },
      { from: "global_review", to: "effect_prior", kind: "phase_advance" },
      { from: "effect_prior", to: "repair_barrier", kind: "repair_transition" },
      { from: "effect_prior", to: "global_prior_review", kind: "repair_transition" },
      { from: "effect_prior", to: "done", kind: "phase_advance" },
      { from: "repair_barrier", to: "global_prior_review", kind: "repair_transition" },
      { from: "repair_barrier", to: "done", kind: "repair_transition" },
      { from: "global_prior_review", to: "done", kind: "phase_advance" },
    ]);
  });

  it("routes representative skip edges outside every non-endpoint section card", () => {
    const cases: Array<[Stage4SectionId, Stage4SectionId]> = [
      ["global_review", "correlation_prior"],
      ["global_review", "effect_prior"],
      ["measurement_prior", "done"],
      ["effect_prior", "repair_barrier"],
      ["correlation_prior", "global_prior_review"],
    ];

    for (const [from, to] of cases) {
      const points = routeStage4SectionEdge(from, to);
      expect(points.length).toBeGreaterThanOrEqual(2);

      for (let index = 1; index < points.length; index++) {
        const start = points[index - 1]!;
        const end = points[index]!;

        for (const sectionId of [
          "model_decisions",
          "global_review",
          "measurement_prior",
          "dynamics_prior",
          "effect_prior",
          "correlation_prior",
          "repair_barrier",
          "global_prior_review",
          "done",
        ] satisfies Stage4SectionId[]) {
          if (sectionId === from || sectionId === to) continue;
          expect(segmentIntersectsRect(start, end, getStage4SectionRect(sectionId))).toBe(false);
        }
      }
    }
  });
});
