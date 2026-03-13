import { afterEach, describe, expect, it } from "vitest";
import { getMockFixture, isMockMode, simulatePipelineEvents } from "./mock-provider";

function unsetEnv(key: string) {
  Reflect.deleteProperty(process.env, key);
}

describe("isMockMode", () => {
  const original = process.env.NEXT_PUBLIC_MOCK_DATA;

  afterEach(() => {
    if (original !== undefined) {
      process.env.NEXT_PUBLIC_MOCK_DATA = original;
    } else {
      unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    }
  });

  it("returns false when env var is unset", () => {
    unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    expect(isMockMode()).toBe(false);
  });

  it("returns false when env var is empty string", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "";
    expect(isMockMode()).toBe(false);
  });

  it("returns false when env var is 'false'", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "false";
    expect(isMockMode()).toBe(false);
  });

  it("returns true when env var is 'true'", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "true";
    expect(isMockMode()).toBe(true);
  });

  it("returns true when env var is a fixture name", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "doctolib";
    expect(isMockMode()).toBe(true);
  });
});

describe("getMockFixture", () => {
  const original = process.env.NEXT_PUBLIC_MOCK_DATA;

  afterEach(() => {
    if (original !== undefined) {
      process.env.NEXT_PUBLIC_MOCK_DATA = original;
    } else {
      unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    }
  });

  it("returns 'DEFAULT' when env var is unset", () => {
    unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    expect(getMockFixture()).toBe("DEFAULT");
  });

  it("returns uppercase fixture userId from env var", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "doctolib";
    expect(getMockFixture()).toBe("DOCTOLIB");
  });

  it("returns 'DEFAULT' when env var is 'true'", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "true";
    expect(getMockFixture()).toBe("DEFAULT");
  });

  it("returns 'DEFAULT' when env var is empty", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "";
    expect(getMockFixture()).toBe("DEFAULT");
  });
});

describe("simulatePipelineEvents", () => {
  it("calls onStageStart and onStageComplete for each stage synchronously", () => {
    const starts: string[] = [];
    const completes: string[] = [];

    simulatePipelineEvents({
      onStageStart: (id) => starts.push(id),
      onStageComplete: (id) => completes.push(id),
    });

    expect(starts.length).toBe(10); // 10 stages
    expect(completes.length).toBe(10);
    expect(starts).toContain("stage-0");
    expect(starts).toContain("stage-6");
    expect(completes).toContain("stage-0");
    expect(completes).toContain("stage-6");
  });

  it("fires onStageStart before onStageComplete for each stage", () => {
    const events: Array<{ type: string; id: string }> = [];

    simulatePipelineEvents({
      onStageStart: (id) => events.push({ type: "start", id }),
      onStageComplete: (id) => events.push({ type: "complete", id }),
    });

    // For each stage, start should come before complete
    for (const id of ["stage-0", "stage-1a", "stage-6"]) {
      const startIdx = events.findIndex((e) => e.type === "start" && e.id === id);
      const completeIdx = events.findIndex((e) => e.type === "complete" && e.id === id);
      expect(startIdx).toBeLessThan(completeIdx);
    }
  });

  it("returns a no-op cleanup function", () => {
    const cleanup = simulatePipelineEvents({
      onStageStart: () => {},
      onStageComplete: () => {},
    });

    expect(typeof cleanup).toBe("function");
    cleanup(); // should not throw
  });
});
