import type { ArtifactViewId } from "@nof1-causal-lab/api-types";
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

  it.each([
    [undefined, false],
    ["", false],
    ["false", false],
    ["true", true],
    ["demo_health", true],
  ])("interprets NEXT_PUBLIC_MOCK_DATA=%s as mock mode %s", (value, expected) => {
    if (value === undefined) {
      unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    } else {
      process.env.NEXT_PUBLIC_MOCK_DATA = value;
    }

    expect(isMockMode()).toBe(expected);
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

  it.each([
    [undefined, "DEFAULT"],
    ["", "DEFAULT"],
    ["true", "DEFAULT"],
    ["demo_health", "DEMO"],
  ])("maps NEXT_PUBLIC_MOCK_DATA=%s to fixture %s", (value, expected) => {
    if (value === undefined) {
      unsetEnv("NEXT_PUBLIC_MOCK_DATA");
    } else {
      process.env.NEXT_PUBLIC_MOCK_DATA = value;
    }

    expect(getMockFixture()).toBe(expected);
  });
});

describe("simulatePipelineEvents", () => {
  it("emits paired start and complete callbacks in machine order", () => {
    const events: Array<{ type: string; id: string }> = [];
    const transitionOrder: ArtifactViewId[] = ["raw_data", "latent_structure", "measurements"];

    const cleanup = simulatePipelineEvents(
      {
        onTransitionStart: (id) => events.push({ type: "start", id }),
        onTransitionComplete: (id) => events.push({ type: "complete", id }),
      },
      transitionOrder,
    );

    expect(events).toHaveLength(transitionOrder.length * 2);
    for (let index = 0; index < events.length; index += 2) {
      expect(events[index]?.type).toBe("start");
      expect(events[index + 1]?.type).toBe("complete");
      expect(events[index]?.id).toBe(events[index + 1]?.id);
    }
    expect(events.filter((event) => event.type === "start").map((event) => event.id)).toEqual(
      transitionOrder,
    );
    expect(new Set(events.map((event) => event.id)).size).toBe(transitionOrder.length);
    cleanup();
  });
});
