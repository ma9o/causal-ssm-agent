"use client";

import { cn } from "@/lib/utils";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { Handle, type NodeProps, Position } from "@xyflow/react";
import { CheckCircle2, Loader2, RotateCcw } from "lucide-react";
import { memo } from "react";

export type Stage4BlockStatus = "pending" | "accepted" | "reopened" | "inactive";

export interface Stage4NodeStatusItem {
  id: string;
  label: string;
  status: Stage4BlockStatus;
  isActive?: boolean;
  inRepairScope?: boolean;
}

export interface Stage4BlockNodeData {
  id: string;
  kind: string;
  label: string;
  phase: string;
  status?: Stage4BlockStatus;
  isActive?: boolean;
  sectionLabel?: string;
  totalCount?: number;
  acceptedCount?: number;
  reopenedCount?: number;
  statusItems?: Stage4NodeStatusItem[];
  detailLabel?: string;
  tooltipText?: string;
  minHeight?: number;
}

function getFrameClass(status: Stage4BlockStatus | undefined, isActive: boolean): string {
  if (isActive) return "border-primary bg-primary/[0.04] ring-1 ring-primary/20 shadow-md";
  switch (status) {
    case "accepted":
      return "border-emerald-500/40 bg-emerald-500/[0.04]";
    case "reopened":
      return "border-amber-500/40 bg-amber-500/[0.05]";
    case "inactive":
      return "border-border/50 bg-card/60 opacity-70";
    default:
      return "border-border bg-card";
  }
}

function getDotClass(item: Stage4NodeStatusItem): string {
  if (item.isActive) return "bg-primary ring-2 ring-primary/20";
  switch (item.status) {
    case "accepted":
      return "bg-emerald-500";
    case "reopened":
      return "bg-amber-500";
    case "inactive":
      return "bg-border/60";
    default:
      return item.inRepairScope ? "bg-amber-300" : "bg-muted";
  }
}

function StatusIcon({
  status,
  isActive,
}: {
  status: Stage4BlockStatus | undefined;
  isActive: boolean;
}) {
  if (isActive) return <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />;
  if (status === "reopened") return <RotateCcw className="h-3.5 w-3.5 text-amber-500" />;
  if (status === "accepted") return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />;
  return <span className="h-2.5 w-2.5 rounded-full bg-muted-foreground/40" />;
}

function Stage4BlockNodeInner({ data }: NodeProps) {
  const d = data as unknown as Stage4BlockNodeData;
  const isActive = d.isActive ?? false;
  const acceptedCount = d.acceptedCount ?? 0;
  const totalCount = d.totalCount ?? 0;
  const reopenedCount = d.reopenedCount ?? 0;

  return (
    <div
      className={cn(
        "min-w-[320px] max-w-[320px] rounded-xl border px-4 py-3 shadow-sm transition-all duration-300",
        getFrameClass(d.status, isActive),
      )}
      style={d.minHeight ? { minHeight: d.minHeight } : undefined}
    >
      <Handle id="top-target" type="target" position={Position.Top} className="!h-2 !w-2 !opacity-0" />
      <Handle id="right-target" type="target" position={Position.Right} className="!h-2 !w-2 !opacity-0" />
      <Handle id="bottom-target" type="target" position={Position.Bottom} className="!h-2 !w-2 !opacity-0" />
      <Handle id="left-target" type="target" position={Position.Left} className="!h-2 !w-2 !opacity-0" />

      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 space-y-1">
          <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
            <span>{d.sectionLabel ?? d.kind}</span>
            {d.tooltipText && <span className="pointer-events-auto"><StatTooltip explanation={d.tooltipText} /></span>}
          </div>
          <div className="flex items-start gap-2">
            <StatusIcon status={d.status} isActive={isActive} />
            <div className="min-w-0">
              <div className="truncate text-[13px] font-semibold leading-tight text-foreground">
                {d.label}
              </div>
              {d.detailLabel && (
                <div className="truncate pt-0.5 text-[11px] text-muted-foreground">
                  {d.detailLabel}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          {totalCount > 0 && (
            <span className="rounded-full border border-border/70 px-2 py-0.5 text-[10px] font-medium tabular-nums text-foreground">
              {acceptedCount}/{totalCount}
            </span>
          )}
        </div>
      </div>

      {d.statusItems && d.statusItems.length > 0 && (
        <div className="mt-3">
          <div className="flex flex-wrap gap-1">
            {d.statusItems.map((item) => (
              <span
                key={item.id}
                title={item.label}
                className={cn(
                  "h-2.5 rounded-full transition-all",
                  item.isActive ? "w-5" : "w-2.5",
                  getDotClass(item),
                )}
              />
            ))}
          </div>
        </div>
      )}

      <Handle id="top-source" type="source" position={Position.Top} className="!h-2 !w-2 !opacity-0" />
      <Handle id="right-source" type="source" position={Position.Right} className="!h-2 !w-2 !opacity-0" />
      <Handle id="bottom-source" type="source" position={Position.Bottom} className="!h-2 !w-2 !opacity-0" />
      <Handle id="left-source" type="source" position={Position.Left} className="!h-2 !w-2 !opacity-0" />
    </div>
  );
}

export const Stage4BlockNode = memo(Stage4BlockNodeInner);
