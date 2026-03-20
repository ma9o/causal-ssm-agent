import { STAGES, STAGE_IDS } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import { describe, expect, it } from "vitest";
import { getStageForPrefectRunName } from "./stages";

describe("STAGE_IDS", () => {
  it("has 10 stages", () => {
    expect(STAGE_IDS).toHaveLength(10);
  });

  it("contains all expected stage IDs", () => {
    const expected = [
      "stage-0",
      "stage-1a",
      "stage-1b",
      "stage-2",
      "stage-3",
      "stage-4",
      "stage-4b",
      "stage-5a",
      "stage-5b",
      "stage-6",
    ];
    expect([...STAGE_IDS]).toEqual(expected);
  });

  it("has no duplicates", () => {
    const unique = new Set(STAGE_IDS);
    expect(unique.size).toBe(STAGE_IDS.length);
  });
});

describe("STAGES metadata", () => {
  it("has one entry per STAGE_ID", () => {
    expect(STAGES).toHaveLength(STAGE_IDS.length);
  });

  it("STAGES ids match STAGE_IDS in order", () => {
    const stageIds = STAGES.map((s) => s.id);
    expect(stageIds).toEqual([...STAGE_IDS]);
  });

  it("every stage has a non-empty label", () => {
    for (const stage of STAGES) {
      expect(stage.label.length).toBeGreaterThan(0);
    }
  });

  it("every stage has a non-empty number", () => {
    for (const stage of STAGES) {
      expect(stage.number.length).toBeGreaterThan(0);
    }
  });

  it("every stage has a non-empty prefectFlowName", () => {
    for (const stage of STAGES) {
      expect(stage.prefectFlowName.length).toBeGreaterThan(0);
    }
  });

  it("every stage has a non-empty loadingHint", () => {
    for (const stage of STAGES) {
      expect(stage.loadingHint.length).toBeGreaterThan(0);
    }
  });

  it("every stage has a non-empty description", () => {
    for (const stage of STAGES) {
      expect(stage.description.length).toBeGreaterThan(0);
    }
  });

  it("prefectFlowName values are unique", () => {
    const names = STAGES.map((s) => s.prefectFlowName);
    expect(new Set(names).size).toBe(names.length);
  });

  it("only stage-1b has a hard gate", () => {
    const gated = STAGES.filter((s) => s.hasGate).map((s) => s.id);
    expect(gated).toEqual(["stage-1b"]);
  });

  it("stage numbers are in expected order", () => {
    const numbers = STAGES.map((s) => s.number);
    expect(numbers).toEqual(["0", "1a", "1b", "2", "3", "4", "4b", "5a", "5b", "6"]);
  });

  it("stage IDs follow stage-{number} pattern", () => {
    for (const stage of STAGES) {
      expect(stage.id).toBe(`stage-${stage.number}` as StageId);
    }
  });
});

describe("getStageForPrefectRunName", () => {
  it("matches exact stage flow names", () => {
    expect(getStageForPrefectRunName("stage-4-flow")?.id).toBe("stage-4");
  });

  it("prefers the longest matching prefix when stage names overlap", () => {
    expect(getStageForPrefectRunName("stage-4b-flow")?.id).toBe("stage-4b");
    expect(getStageForPrefectRunName("stage-4b-flow-retry-1")?.id).toBe("stage-4b");
  });

  it("returns undefined for unrelated run names", () => {
    expect(getStageForPrefectRunName("extract-chunk-0")).toBeUndefined();
  });
});
