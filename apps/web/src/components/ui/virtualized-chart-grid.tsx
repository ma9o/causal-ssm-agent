"use client";

import { useVirtualizer } from "@tanstack/react-virtual";
import { type ReactNode, useEffect, useRef, useState } from "react";

function getColumns(containerWidth: number): number {
  if (containerWidth >= 800) return 3;
  if (containerWidth >= 500) return 2;
  return 1;
}

interface VirtualizedChartGridProps<T> {
  items: T[];
  /** Estimated height of one row (including row gap) in px */
  estimateRowHeight: number;
  /** Max height of the scroll viewport in px */
  maxHeight: number;
  renderItem: (item: T) => ReactNode;
  keyExtractor: (item: T) => string;
}

export function VirtualizedChartGrid<T>({
  items,
  estimateRowHeight,
  maxHeight,
  renderItem,
  keyExtractor,
}: VirtualizedChartGridProps<T>) {
  "use no memo"; // TODO: remove when TanStack Virtual supports React Compiler
  const scrollRef = useRef<HTMLDivElement>(null);
  const [columns, setColumns] = useState(1);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? 0;
      setColumns(getColumns(width));
    });
    observer.observe(el);
    setColumns(getColumns(el.clientWidth));
    return () => observer.disconnect();
  }, []);

  const rowCount = Math.ceil(items.length / columns);

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => estimateRowHeight,
    overscan: 2,
  });

  return (
    <div ref={scrollRef} className="overflow-y-auto" style={{ maxHeight }}>
      <div
        style={{
          height: virtualizer.getTotalSize(),
          position: "relative",
          width: "100%",
        }}
      >
        {virtualizer.getVirtualItems().map((vRow) => {
          const startIdx = vRow.index * columns;
          const rowItems = items.slice(startIdx, startIdx + columns);
          return (
            <div
              key={vRow.key}
              data-index={vRow.index}
              ref={virtualizer.measureElement}
              className="pb-4"
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                transform: `translateY(${vRow.start}px)`,
                display: "grid",
                gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
                gap: "1rem",
              }}
            >
              {rowItems.map((item) => (
                <div key={keyExtractor(item)}>{renderItem(item)}</div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
