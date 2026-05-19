// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { PrefectLogEntry } from "@/lib/prefect-log-client";
import { isNearLogTail, VirtualizedLogList } from "./virtualized-log-list";

declare global {
  var IS_REACT_ACT_ENVIRONMENT: boolean | undefined;
}

const scrollToIndex = vi.fn();

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getTotalSize: () => count * 20,
    getVirtualItems: () =>
      Array.from({ length: count }, (_, index) => ({
        index,
        start: index * 20,
      })),
    measureElement: () => undefined,
    scrollToIndex,
  }),
}));

function makeLog(id: string): PrefectLogEntry {
  return {
    id,
    created: "2026-04-01T13:00:00.000Z",
    name: "prefect.flow_runs",
    level: 20,
    message: `log-${id}`,
    timestamp: "2026-04-01T13:00:00.000Z",
    flow_run_id: "run-1",
    task_run_id: null,
  };
}

describe("isNearLogTail", () => {
  it("treats positions near the bottom as pinned", () => {
    expect(isNearLogTail(952, 1200, 200)).toBe(true);
    expect(isNearLogTail(900, 1200, 200)).toBe(false);
  });
});

describe("VirtualizedLogList", () => {
  const originalRaf = globalThis.requestAnimationFrame;
  const originalCancelRaf = globalThis.cancelAnimationFrame;

  let container: HTMLDivElement | null = null;
  let root: Root | null = null;

  function getContainer(): HTMLDivElement {
    if (!container) {
      throw new Error("Test container has not been initialized");
    }
    return container;
  }

  function getRoot(): Root {
    if (!root) {
      throw new Error("React root has not been initialized");
    }
    return root;
  }

  beforeEach(() => {
    scrollToIndex.mockReset();
    // React 19 warns unless the test env opts into act-aware updates.
    globalThis.IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    globalThis.requestAnimationFrame = ((callback: FrameRequestCallback) => {
      callback(0);
      return 1;
    }) as typeof requestAnimationFrame;
    globalThis.cancelAnimationFrame = vi.fn();
  });

  afterEach(() => {
    if (root) {
      act(() => {
        root?.unmount();
      });
    }
    container?.remove();
    globalThis.IS_REACT_ACT_ENVIRONMENT = false;
    globalThis.requestAnimationFrame = originalRaf;
    globalThis.cancelAnimationFrame = originalCancelRaf;
  });

  it("pins the live view to the tail when new logs arrive", () => {
    act(() => {
      getRoot().render(
        createElement(VirtualizedLogList, {
          logs: [makeLog("1"), makeLog("2")],
          emptyMessage: "Waiting for logs...",
          autoScroll: true,
        }),
      );
    });

    const scrollContainer = getContainer().querySelector(".max-h-64") as HTMLDivElement | null;
    expect(scrollContainer).toBeTruthy();
    if (!scrollContainer) {
      throw new Error("Could not find virtualized log scroll container");
    }

    Object.defineProperty(scrollContainer, "scrollHeight", {
      configurable: true,
      get: () => 2400,
    });
    Object.defineProperty(scrollContainer, "clientHeight", {
      configurable: true,
      get: () => 256,
    });
    scrollContainer.scrollTop = 2144;

    act(() => {
      scrollContainer.dispatchEvent(new Event("scroll"));
    });

    act(() => {
      getRoot().render(
        createElement(VirtualizedLogList, {
          logs: [makeLog("1"), makeLog("2"), makeLog("3")],
          emptyMessage: "Waiting for logs...",
          autoScroll: true,
        }),
      );
    });

    expect(scrollToIndex).toHaveBeenLastCalledWith(2, { align: "end" });
    expect(scrollContainer.scrollTop).toBe(2400);
  });
});
