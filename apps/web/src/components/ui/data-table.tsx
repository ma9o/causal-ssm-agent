"use client";

import { useMemo, useRef } from "react";
import { useVirtualizer } from "@tanstack/react-virtual";
import { cn } from "@/lib/utils/cn";
import { useTableKeyboardNav } from "./use-table-keyboard-nav";

interface DataTableProps<T extends object> {
  rows: T[];
  maxHeight?: string;
}

const ROW_HEIGHT = 28;

export function DataTable<T extends object>({ rows, maxHeight = "max-h-64" }: DataTableProps<T>) {
  if (rows.length === 0) return null;

  const columns = useMemo(() => {
    const firstRow = rows[0] as Record<string, unknown>;
    const allKeys = Object.keys(firstRow);
    return allKeys.filter((key) =>
      rows.some((row) => (row as Record<string, unknown>)[key] != null),
    );
  }, [rows]);

  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: rows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 10,
  });

  const { focusedRowIndex, containerProps } = useTableKeyboardNav(rows.length);

  return (
    <div
      ref={parentRef}
      className={cn(maxHeight, "overflow-y-auto rounded-md border")}
      {...containerProps}
    >
      {/* Sticky header */}
      <div
        className="sticky top-0 z-10 flex border-b bg-background"
        role="row"
      >
        {columns.map((col) => (
          <div
            key={col}
            className="flex-1 min-w-0 py-1 px-3 text-xs font-medium text-muted-foreground capitalize truncate"
            role="columnheader"
          >
            {col.replace(/_/g, " ")}
          </div>
        ))}
      </div>

      {/* Virtualized body */}
      <div
        style={{ height: virtualizer.getTotalSize(), position: "relative" }}
        role="rowgroup"
      >
        {virtualizer.getVirtualItems().map((vi) => {
          const row = rows[vi.index] as Record<string, unknown>;
          return (
            <div
              key={vi.index}
              className={cn(
                "absolute left-0 right-0 flex border-b border-border/40 hover:bg-muted/50",
                focusedRowIndex === vi.index && "ring-2 ring-ring ring-inset",
              )}
              style={{
                height: vi.size,
                transform: `translateY(${vi.start}px)`,
              }}
              role="row"
            >
              {columns.map((col) => (
                <div
                  key={col}
                  className="flex-1 min-w-0 py-1 px-3 text-xs text-muted-foreground truncate leading-5"
                  role="gridcell"
                >
                  {row[col] == null
                    ? ""
                    : typeof row[col] === "boolean"
                      ? row[col]
                        ? "true"
                        : "false"
                      : String(row[col])}
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
