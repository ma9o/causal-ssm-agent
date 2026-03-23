import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it } from "vitest";
import {
  getStage2WorkerQueryKey,
  getStage2WorkerQueryKeyPrefix,
  type Stage2Worker,
} from "./use-stage2-workers";

const RUN1_WORKERS: Stage2Worker[] = [
  { id: "worker-0", name: "extract-chunk-0", state: "completed", nLlmCalls: 2, completedAt: 1 },
];

const RUN2_WORKERS: Stage2Worker[] = [
  { id: "worker-0", name: "extract-chunk-0", state: "running" },
];

describe("getStage2WorkerQueryKey", () => {
  it("scopes stage-2 worker cache entries by root flow run", () => {
    expect(getStage2WorkerQueryKey("user-123", "run-1")).toEqual([
      "pipeline",
      "user-123",
      "stage2-workers",
      "run-1",
    ]);
    expect(getStage2WorkerQueryKey("user-123", "run-2")).toEqual([
      "pipeline",
      "user-123",
      "stage2-workers",
      "run-2",
    ]);
  });

  it("supports clearing all stage-2 worker caches for a user across runs", () => {
    const queryClient = new QueryClient();

    queryClient.setQueryData(getStage2WorkerQueryKey("user-123", "run-1"), RUN1_WORKERS);
    queryClient.setQueryData(getStage2WorkerQueryKey("user-123", "run-2"), RUN2_WORKERS);
    queryClient.setQueryData(getStage2WorkerQueryKey("other-user", "run-9"), RUN1_WORKERS);

    queryClient.removeQueries({ queryKey: getStage2WorkerQueryKeyPrefix("user-123") });

    expect(queryClient.getQueryData(getStage2WorkerQueryKey("user-123", "run-1"))).toBeUndefined();
    expect(queryClient.getQueryData(getStage2WorkerQueryKey("user-123", "run-2"))).toBeUndefined();
    expect(queryClient.getQueryData(getStage2WorkerQueryKey("other-user", "run-9"))).toEqual(
      RUN1_WORKERS,
    );
  });
});
