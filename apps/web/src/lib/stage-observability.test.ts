import { describe, expect, it } from "vitest";
import type { AnalysisStageRun } from "./api/analysis";
import {
  buildPrefectSubscriptionKey,
  buildStageLogScopePath,
  getStageLogQueryScopeKey,
  getStageLogScopePolicy,
  getStageRuntimeInitialLogFlowRunIds,
  shouldRefreshStageLogScope,
  toStageRuntimeRef,
} from "./stage-observability";

function makeStageRun(overrides: Partial<AnalysisStageRun> = {}): AnalysisStageRun {
  return {
    ownerRootFlowRunId: "root-123",
    stageSubflowRunId: "subflow-123",
    initialLogFlowRunIds: ["subflow-123"],
    execution: null,
    ...overrides,
  };
}

describe("stage observability", () => {
  it("uses a dynamic child-flow log scope policy for stage 2 only", () => {
    expect(getStageLogScopePolicy("stage-2")).toBe("subflow-with-children");
    expect(getStageLogScopePolicy("stage-4")).toBe("subflow");
  });

  it("derives a stable log query key from runtime identity instead of dynamic flow scope", () => {
    const runtime = toStageRuntimeRef(
      makeStageRun({
        initialLogFlowRunIds: ["subflow-123", "worker-1", "worker-2"],
      }),
    );

    expect(getStageLogQueryScopeKey(runtime)).toBe("subflow-123");
  });

  it("falls back to the stage subflow when explicit log flow ids are absent", () => {
    expect(
      getStageRuntimeInitialLogFlowRunIds(
        makeStageRun({
          initialLogFlowRunIds: [],
        }),
      ),
    ).toEqual(["subflow-123"]);
  });

  it("builds the generic stage log-scope route path", () => {
    expect(buildStageLogScopePath("user-123", "stage-2", "subflow-123")).toBe(
      "/api/analysis/user-123/stages/stage-2/log-scope?stageSubflowRunId=subflow-123",
    );
  });

  it("refreshes dynamic log scopes only while the stage is running", () => {
    expect(shouldRefreshStageLogScope("stage-2", true, "subflow-123")).toBe(true);
    expect(shouldRefreshStageLogScope("stage-2", false, "subflow-123")).toBe(false);
    expect(shouldRefreshStageLogScope("stage-4", true, "subflow-123")).toBe(false);
  });

  it("normalizes subscription keys from flow run ids", () => {
    expect(buildPrefectSubscriptionKey([" subflow-123 ", "worker-1", "worker-1"])).toBe(
      "subflow-123|worker-1",
    );
  });
});
