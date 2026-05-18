"use client";

import { StatTooltip } from "@/components/ui/stat-tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import {
  type ColumnDef,
  type Row,
  type SortingState,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ChevronDown, ChevronUp, ChevronsUpDown, Search } from "lucide-react";
import { type ReactNode, useMemo, useRef, useState } from "react";
import { useTableKeyboardNav } from "./use-table-keyboard-nav";

// ---------- Column meta typing ----------
declare module "@tanstack/react-table" {
  interface ColumnMeta<TData, TValue> {
    align?: "left" | "center" | "right";
    mono?: boolean;
    /** Thresholding function: "fail" → red bg, "warn" → orange bg, undefined → default. */
    severity?: (value: TValue, row: TData) => "fail" | "warn" | undefined;
  }
}

function severityClass(level: "fail" | "warn" | undefined): string | undefined {
  if (level === "fail") return "bg-destructive/10";
  if (level === "warn") return "bg-warning/15";
  return undefined;
}

// ---------- HeaderWithTooltip helper ----------
export function HeaderWithTooltip({
  label,
  tooltip,
  className,
}: {
  label: string;
  tooltip: string;
  className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1", className)}>
      {label}
      <StatTooltip explanation={tooltip} />
    </span>
  );
}

// ---------- Flat item for virtualized rendering ----------
type FlatItem<TData> =
  | { kind: "group-header"; groupKey: string; rows: TData[] }
  | { kind: "row"; row: Row<TData> }
  | { kind: "expanded-row"; row: Row<TData> };

// ---------- InfoTable ----------
const GROUP_HEADER_HEIGHT = 36;

interface InfoTableProps<TData> {
  columns: ColumnDef<TData, unknown>[];
  data: TData[];
  sorting?: boolean;
  filtering?: boolean;
  compact?: boolean;
  maxHeight?: string;
  estimateRowHeight?: number;
  groupBy?: (row: TData) => string;
  renderGroupHeader?: (groupKey: string, rows: TData[]) => ReactNode;
  rowClassName?: (row: TData, index: number) => string | undefined;
  isRowExpanded?: (row: TData) => boolean;
  renderExpandedRow?: (row: TData) => ReactNode;
}

export function InfoTable<TData>({
  columns,
  data,
  sorting: enableSorting = true,
  filtering: enableFiltering = true,
  compact: isCompact = false,
  maxHeight = "max-h-[32rem]",
  estimateRowHeight = 40,
  groupBy,
  renderGroupHeader,
  rowClassName,
  isRowExpanded,
  renderExpandedRow,
}: InfoTableProps<TData>) {
  "use no memo"; // TODO: remove when TanStack Table supports React Compiler
  const [sortingState, setSortingState] = useState<SortingState>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const parentRef = useRef<HTMLDivElement>(null);

  // Pre-filter data before passing to TanStack Table
  const filteredData = useMemo(() => {
    if (!searchQuery) return data;
    const search = searchQuery.toLowerCase();
    return data.filter((row) => JSON.stringify(row).toLowerCase().includes(search));
  }, [data, searchQuery]);

  const table = useReactTable({
    data: filteredData,
    columns,
    state: enableSorting ? { sorting: sortingState } : undefined,
    onSortingChange: enableSorting ? setSortingState : undefined,
    getCoreRowModel: getCoreRowModel(),
    ...(enableSorting && { getSortedRowModel: getSortedRowModel() }),
  });

  const rows = table.getRowModel().rows;

  // Build flat list: group headers interleaved with data rows
  const flatItems = useMemo<FlatItem<TData>[]>(() => {
    if (!groupBy) {
      const items: FlatItem<TData>[] = [];
      for (const row of rows) {
        items.push({ kind: "row", row });
        if (isRowExpanded?.(row.original)) {
          items.push({ kind: "expanded-row", row });
        }
      }
      return items;
    }
    const map = new Map<string, typeof rows>();
    for (const row of rows) {
      const key = groupBy(row.original);
      const list = map.get(key) ?? [];
      list.push(row);
      map.set(key, list);
    }
    const items: FlatItem<TData>[] = [];
    for (const [groupKey, groupRows] of map) {
      if (renderGroupHeader) {
        items.push({
          kind: "group-header",
          groupKey,
          rows: groupRows.map((r) => r.original),
        });
      }
      for (const row of groupRows) {
        items.push({ kind: "row", row });
        if (isRowExpanded?.(row.original)) {
          items.push({ kind: "expanded-row", row });
        }
      }
    }
    return items;
  }, [groupBy, renderGroupHeader, rows, isRowExpanded]);

  // For small tables, render every row by setting overscan to the full count.
  // With variable-height cells (e.g. wrapping text) the browser's auto table-layout
  // recomputes column widths from visible cells, so virtualizing rows in/out can
  // oscillate column widths → row heights → measurements in an infinite feedback loop.
  // Keeping all rows mounted breaks the loop; virtualization still helps for large tables.
  const virtualizer = useVirtualizer({
    count: flatItems.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) =>
      flatItems[index].kind === "group-header" ? GROUP_HEADER_HEIGHT : estimateRowHeight,
    overscan: flatItems.length <= 100 ? flatItems.length : 5,
  });

  const { focusedRowIndex, containerProps } = useTableKeyboardNav(rows.length);

  const virtualItems = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();

  // Spacer heights for the padding approach (keeps <table> semantics)
  const paddingTop = virtualItems.length > 0 ? virtualItems[0].start : 0;
  const paddingBottom =
    virtualItems.length > 0 ? totalSize - virtualItems[virtualItems.length - 1].end : 0;

  const isFiltered = searchQuery.length > 0;

  return (
    <div className="overflow-hidden rounded-md border">
      {enableFiltering && (
        <div className="flex items-center gap-2 border-b px-3 py-1.5">
          <Search className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
          <input
            type="text"
            placeholder={`Search ${data.length} rows\u2026`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="flex-1 bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          />
          {isFiltered && (
            <span className="shrink-0 text-xs text-muted-foreground">
              {filteredData.length} of {data.length}
            </span>
          )}
        </div>
      )}
      <div ref={parentRef} className={cn(maxHeight, "overflow-auto")} {...containerProps}>
        <Table>
          <TableHeader className="sticky top-0 bg-background z-10 shadow-[0_1px_0_var(--border)]">
            {table.getHeaderGroups().map((hg) => (
              <TableRow key={hg.id}>
                {hg.headers.map((header) => {
                  const meta = header.column.columnDef.meta;
                  const alignClass =
                    meta?.align === "right"
                      ? "text-right"
                      : meta?.align === "center"
                        ? "text-center"
                        : "text-left";

                  return (
                    <TableHead
                      key={header.id}
                      className={cn(alignClass)}
                      onClick={
                        enableSorting && header.column.getCanSort()
                          ? header.column.getToggleSortingHandler()
                          : undefined
                      }
                      style={
                        enableSorting && header.column.getCanSort()
                          ? { cursor: "pointer", userSelect: "none" }
                          : undefined
                      }
                      aria-sort={
                        enableSorting && header.column.getIsSorted()
                          ? header.column.getIsSorted() === "asc"
                            ? "ascending"
                            : "descending"
                          : undefined
                      }
                    >
                      <span className="inline-flex items-center gap-1">
                        {header.isPlaceholder
                          ? null
                          : flexRender(header.column.columnDef.header, header.getContext())}
                        {enableSorting &&
                          header.column.getCanSort() &&
                          (header.column.getIsSorted() === "asc" ? (
                            <ChevronUp className="h-3.5 w-3.5" />
                          ) : header.column.getIsSorted() === "desc" ? (
                            <ChevronDown className="h-3.5 w-3.5" />
                          ) : (
                            <ChevronsUpDown className="h-3 w-3 text-muted-foreground/50" />
                          ))}
                      </span>
                    </TableHead>
                  );
                })}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {paddingTop > 0 && (
              <tr>
                <td style={{ height: paddingTop, padding: 0, border: "none" }} />
              </tr>
            )}
            {virtualItems.map((vi) => {
              const item = flatItems[vi.index];
              if (item.kind === "group-header") {
                return (
                  <TableRow
                    key={`group-${item.groupKey}`}
                    className="bg-muted/50 hover:bg-muted/50"
                    data-index={vi.index}
                    ref={virtualizer.measureElement}
                  >
                    <TableCell colSpan={columns.length} className="py-2">
                      {renderGroupHeader?.(item.groupKey, item.rows)}
                    </TableCell>
                  </TableRow>
                );
              }
              if (item.kind === "expanded-row") {
                return (
                  <TableRow
                    key={`expanded-${item.row.id}`}
                    className="bg-muted/20 hover:bg-muted/20"
                    data-index={vi.index}
                    ref={virtualizer.measureElement}
                  >
                    <TableCell colSpan={columns.length} className="p-0">
                      {renderExpandedRow?.(item.row.original)}
                    </TableCell>
                  </TableRow>
                );
              }
              const { row } = item;
              const nextItem = flatItems[vi.index + 1];
              const hasExpandedBelow = nextItem?.kind === "expanded-row";
              return (
                <TableRow
                  key={row.id}
                  className={cn(
                    hasExpandedBelow && "border-b-0",
                    focusedRowIndex === row.index && "ring-2 ring-ring ring-inset",
                    rowClassName?.(row.original, row.index),
                  )}
                  data-index={vi.index}
                  ref={virtualizer.measureElement}
                >
                  {row.getVisibleCells().map((cell) => {
                    const meta = cell.column.columnDef.meta;
                    const alignClass =
                      meta?.align === "right"
                        ? "text-right"
                        : meta?.align === "center"
                          ? "text-center"
                          : "text-left";
                    const sev = meta?.severity?.(cell.getValue(), cell.row.original);

                    return (
                      <TableCell
                        key={cell.id}
                        className={cn(alignClass, meta?.mono && "font-mono", severityClass(sev), isCompact && "py-0.5")}
                      >
                        {flexRender(cell.column.columnDef.cell, cell.getContext())}
                      </TableCell>
                    );
                  })}
                </TableRow>
              );
            })}
            {paddingBottom > 0 && (
              <tr>
                <td style={{ height: paddingBottom, padding: 0, border: "none" }} />
              </tr>
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
