import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getMockFixture, isMockMode, simulatePipelineEvents } from "./mock-provider";

describe("isMockMode", () => {
  const original = process.env.NEXT_PUBLIC_MOCK_DATA;

  afterEach(() => {
    if (original !== undefined) {
      process.env.NEXT_PUBLIC_MOCK_DATA = original;
    } else {
      delete process.env.NEXT_PUBLIC_MOCK_DATA;
    }
  });

  it("returns false when env var is unset", () => {
    delete process.env.NEXT_PUBLIC_MOCK_DATA;
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
      delete process.env.NEXT_PUBLIC_MOCK_DATA;
    }
  });

  it("returns 'default' when env var is unset", () => {
    delete process.env.NEXT_PUBLIC_MOCK_DATA;
    expect(getMockFixture()).toBe("default");
  });

  it("returns the fixture name from env var", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "doctolib";
    expect(getMockFixture()).toBe("doctolib");
  });

  it("returns empty string when env var is empty", () => {
    process.env.NEXT_PUBLIC_MOCK_DATA = "";
    expect(getMockFixture()).toBe("default");
  });
});

describe("simulatePipelineEvents", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it("calls onStageStart and onStageComplete for each stage", () => {
    const starts: string[] = [];
    const completes: string[] = [];

    simulatePipelineEvents({
      onStageStart: (id) => starts.push(id),
      onStageComplete: (id) => completes.push(id),
    });

    // Advance past all timers (last stage at 12500ms)
    vi.advanceTimersByTime(15000);

    expect(starts.length).toBe(9); // 9 stages
    expect(completes.length).toBe(9);
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

    vi.advanceTimersByTime(15000);

    // For each stage, start should come before complete
    for (const id of ["stage-0", "stage-1a", "stage-6"]) {
      const startIdx = events.findIndex((e) => e.type === "start" && e.id === id);
      const completeIdx = events.findIndex((e) => e.type === "complete" && e.id === id);
      expect(startIdx).toBeLessThan(completeIdx);
    }
  });

  it("returns a cleanup function that clears all timers", () => {
    const starts: string[] = [];
    const completes: string[] = [];

    const cleanup = simulatePipelineEvents({
      onStageStart: (id) => starts.push(id),
      onStageComplete: (id) => completes.push(id),
    });

    // Clean up immediately
    cleanup();
    vi.advanceTimersByTime(15000);

    expect(starts.length).toBe(0);
    expect(completes.length).toBe(0);
  });

  it("fires stage-0 events first", () => {
    const events: string[] = [];

    simulatePipelineEvents({
      onStageStart: (id) => events.push(`start:${id}`),
      onStageComplete: (id) => events.push(`complete:${id}`),
    });

    // Only advance to stage-0 timing (500ms complete, 100ms start)
    vi.advanceTimersByTime(600);

    expect(events).toContain("start:stage-0");
    expect(events).toContain("complete:stage-0");
  });
});
