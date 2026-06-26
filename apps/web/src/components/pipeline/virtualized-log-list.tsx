"use client";

import { type PrefectLogEntry, logLevelLabel } from "@/lib/prefect-log-client";
import { cn } from "@/lib/utils";
import { useVirtualizer } from "@tanstack/react-virtual";
import { useEffect, useRef } from "react";

const LEVEL_COLORS: Record<number, string> = {
  10: "text-muted-foreground/50",
  20: "text-muted-foreground",
  30: "text-amber-500",
  40: "text-red-500",
  50: "text-red-600 font-semibold",
};

const ESTIMATED_LOG_ROW_HEIGHT = 20;
const AUTO_SCROLL_BOTTOM_THRESHOLD_PX = 48;

export function isNearLogTail(
  scrollTop: number,
  scrollHeight: number,
  clientHeight: number,
  thresholdPx = AUTO_SCROLL_BOTTOM_THRESHOLD_PX,
): boolean {
  return scrollHeight - (scrollTop + clientHeight) <= thresholdPx;
}

function LogLine({ entry }: { entry: PrefectLogEntry }) {
  const ts = new Date(entry.timestamp).toLocaleTimeString();
  const level = logLevelLabel(entry.level);
  const color = LEVEL_COLORS[entry.level] ?? "text-muted-foreground";

  return (
    <div className="flex gap-2 leading-5 hover:bg-muted/30">
      <span className="shrink-0 text-muted-foreground/40 select-none">{ts}</span>
      <span className={cn("shrink-0 w-12 text-right", color)}>{level}</span>
      <span className={cn("break-all", entry.level >= 40 && "text-red-500")}>{entry.message}</span>
    </div>
  );
}

export function VirtualizedLogList({
  logs,
  emptyMessage,
  className,
  autoScroll = true,
}: {
  logs: PrefectLogEntry[];
  emptyMessage: string;
  className?: string;
  autoScroll?: boolean;
}) {
  "use no memo"; // TODO: remove when TanStack Virtual supports React Compiler
  const parentRef = useRef<HTMLDivElement>(null);
  const pinnedToTailRef = useRef(true);

  const virtualizer = useVirtualizer({
    count: logs.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ESTIMATED_LOG_ROW_HEIGHT,
    overscan: 12,
  });

  useEffect(() => {
    const element = parentRef.current;
    if (!element) {
      return;
    }

    if (!autoScroll) {
      pinnedToTailRef.current = false;
      return;
    }

    const updatePinnedState = () => {
      pinnedToTailRef.current = isNearLogTail(
        element.scrollTop,
        element.scrollHeight,
        element.clientHeight,
      );
    };

    updatePinnedState();
    element.addEventListener("scroll", updatePinnedState, { passive: true });

    return () => {
      element.removeEventListener("scroll", updatePinnedState);
    };
  }, [autoScroll]);

  useEffect(() => {
    if (!autoScroll || logs.length === 0 || !pinnedToTailRef.current) {
      return;
    }

    const frame = requestAnimationFrame(() => {
      const element = parentRef.current;
      if (!element) {
        return;
      }

      virtualizer.scrollToIndex(logs.length - 1, { align: "end" });
      element.scrollTop = element.scrollHeight;
      pinnedToTailRef.current = true;
    });

    return () => cancelAnimationFrame(frame);
  }, [autoScroll, logs.length, virtualizer]);

  return (
    <div
      ref={parentRef}
      className={cn("max-h-64 overflow-y-auto rounded-md font-mono text-[11px]", className)}
    >
      {logs.length === 0 ? (
        <p className="py-2 text-center text-muted-foreground/50">{emptyMessage}</p>
      ) : (
        <div className="relative w-full" style={{ height: virtualizer.getTotalSize() }}>
          {virtualizer.getVirtualItems().map((item) => {
            const entry = logs[item.index];
            return (
              <div
                key={entry.id}
                data-index={item.index}
                ref={(node) => {
                  if (node) {
                    virtualizer.measureElement(node);
                  }
                }}
                className="absolute left-0 right-0 top-0"
                style={{ transform: `translateY(${item.start}px)` }}
              >
                <LogLine entry={entry} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
