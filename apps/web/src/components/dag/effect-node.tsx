"use client";

import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Construct, Indicator } from "@nof1-causal-lab/api-types";
import { Handle, type NodeProps, Position } from "@xyflow/react";
import { Lock, Star } from "lucide-react";
import { memo } from "react";
import type { NodeAnimPhase } from "./intervention-dag-types";
import { Sparkline } from "./sparkline";

interface EffectNodeData extends Construct {
  rung?: 2 | 3;
  indicators?: Indicator[];
  animPhase?: NodeAnimPhase;
  effectMagnitude?: number | null;
  startStateValue?: number | null;
  timeIndex?: number;
  timeStepsDays?: number[] | null;
  referenceTimeSeries?: number[] | null;
  comparisonTimeSeries?: number[] | null;
  actionLabelShort?: string | null;
  actionReferenceLabel?: string | null;
}

const DISPLAY_EPSILON = 1e-6;

/** Color class for effect sign. */
function effectColorClass(positive: boolean): string {
  return positive
    ? "text-teal-600 dark:text-teal-400"
    : "text-rose-600 dark:text-rose-400";
}

/** CSS color value for SVG strokes. */
function effectColorVar(positive: boolean): string {
  return positive ? "var(--color-teal-500)" : "var(--color-rose-500)";
}

function EffectNodeInner({ data, selected }: NodeProps) {
  const d = data as unknown as EffectNodeData;
  const indicators = d.indicators ?? [];
  const phase = d.animPhase ?? "idle";
  const effect = d.effectMagnitude;
  const startState = d.startStateValue;

  const isClamped = phase === "clamped";
  const isActive = phase === "active";
  const isStartState = phase === "start_state";
  const isDimmed = phase === "dimmed";
  const isReceiving = phase === "receiving";

  const hasRenderableEffect =
    effect != null && Math.abs(effect) > DISPLAY_EPSILON;
  const showEffect =
    (isActive || isClamped || isReceiving) && hasRenderableEffect;
  const positive = (effect ?? 0) >= 0;

  // ── Sparkline decision ────────────────────────────────────────────
  const timeIndex = d.timeIndex ?? 0;
  const tsDays = d.timeStepsDays;
  const referenceTs = d.referenceTimeSeries;
  const comparisonTs = d.comparisonTimeSeries;
  const actionLabelShort = d.actionLabelShort;
  const actionReferenceLabel = d.actionReferenceLabel;

  const isRung3 = d.rung === 3;
  const hasSeries = referenceTs != null && comparisonTs != null;
  const showSparkline =
    hasSeries &&
    tsDays != null &&
    tsDays.length > 0 &&
    timeIndex > 0 &&
    (isClamped || isActive || isReceiving);

  return (
    <div
      className={cn(
        "relative rounded-lg border-2 shadow-sm transition-all duration-300 cursor-pointer",
        "hover:shadow-md hover:-translate-y-0.5",
        "bg-card",
        isClamped && "border-blue-500 ring-2 ring-blue-500/30",
        isReceiving && "border-teal-400/80",
        isActive &&
          hasRenderableEffect &&
          (positive ? "border-teal-500" : "border-rose-500"),
        isStartState && "border-amber-400 ring-2 ring-amber-400/20",
        isDimmed && "opacity-40",
        phase === "idle" &&
          (d.role === "endogenous"
            ? "border-foreground/65"
            : "border-foreground/35"),
        d.is_outcome &&
          !isClamped &&
          !isActive &&
          !isStartState &&
          "ring-2 ring-foreground/75 ring-offset-1",
        selected && "shadow-lg ring-2 ring-primary ring-offset-2",
      )}
    >
      {/* Tinted overlay */}
      {showEffect && (
        <div
          className={cn(
            "absolute inset-0 rounded-[6px] pointer-events-none transition-opacity duration-500",
            positive ? "bg-teal-500" : "bg-rose-500",
          )}
          style={{ opacity: Math.min(0.18, Math.abs(effect!) * 0.22) }}
        />
      )}

      <Handle
        type="target"
        position={Position.Top}
        className="!bg-muted-foreground !w-2 !h-2"
      />

      <div className="relative px-4 py-3">
        {/* Header */}
        <div className="flex items-center gap-1.5">
          {isClamped && (
            <Lock className="h-3.5 w-3.5 shrink-0 text-blue-500" />
          )}
          <span className="text-sm font-semibold leading-tight">{d.name}</span>
          {d.is_outcome && (
            <Star className="h-3.5 w-3.5 shrink-0 fill-foreground/75 text-foreground/75" />
          )}
        </div>

        {/* Badges */}
        <div className="mt-1.5 flex flex-wrap gap-1">
          <Badge
            variant={d.role === "endogenous" ? "default" : "secondary"}
            className="px-1.5 py-0 text-[10px]"
          >
            {d.role === "endogenous" ? "endo" : "exo"}
          </Badge>
          <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
            {d.temporal_status === "time_varying" ? "varying" : "invariant"}
          </Badge>
        </div>

        {/* ── Effect annotations ─────────────────────────────────── */}

        {showSparkline ? (
          /* Live sparkline + numeric label */
          <div className="mt-2 space-y-0.5">
            <Sparkline
              series={comparisonTs!}
              baselineSeries={referenceTs!}
              visibleCount={timeIndex + 1}
              days={tsDays!}
              color={isClamped ? "var(--color-blue-500)" : effectColorVar(positive)}
            />
            <div
              className={cn(
                "text-[10px] font-mono font-semibold",
                isClamped
                  ? "text-blue-600 dark:text-blue-400"
                  : effectColorClass(positive),
              )}
            >
              {isClamped
                ? (actionLabelShort ?? "intervene")
                : isRung3
                  ? `\u0394 = ${effect! >= 0 ? "+" : ""}${effect!.toFixed(3)}`
                  : `${positive ? "+" : ""}${effect!.toFixed(3)}`}
            </div>
            {isClamped && actionReferenceLabel ? (
              <div className="text-[9px] text-muted-foreground">
                {actionReferenceLabel}
              </div>
            ) : null}
          </div>
        ) : (
          /* Text-only fallback (static mode, early phases, unaffected nodes) */
          <>
            {isClamped && (
              <div className="mt-2 space-y-0.5">
                <div className="text-xs font-mono font-semibold text-blue-600 dark:text-blue-400">
                  {actionLabelShort ?? "intervene"}
                </div>
                {actionReferenceLabel ? (
                  <div className="text-[10px] text-muted-foreground">
                    {actionReferenceLabel}
                  </div>
                ) : null}
              </div>
            )}

            {showEffect && !isClamped && (
              <div className="mt-2 space-y-0.5">
                <div
                  className={cn(
                    "text-xs font-mono font-semibold",
                    effectColorClass(positive),
                  )}
                >
                  {isRung3 ? "\u0394 = " : positive ? "+" : ""}
                  {isRung3
                    ? `${effect! >= 0 ? "+" : ""}${effect!.toFixed(3)}`
                    : effect!.toFixed(3)}
                </div>
              </div>
            )}

            {isStartState && startState != null && (
              <div className="mt-2 text-xs font-mono font-semibold text-amber-600 dark:text-amber-400">
                {"\u03B7"} = {startState.toFixed(2)}
              </div>
            )}
          </>
        )}
      </div>

      {/* Indicators */}
      {indicators.length > 0 && (
        <div className="border-t border-dashed border-border px-3 py-1.5">
          {indicators.map((ind) => (
            <div
              key={ind.name}
              className="flex items-center justify-between gap-2 py-0.5"
            >
              <span className="text-[11px] text-muted-foreground truncate">
                {ind.name}
              </span>
              <span className="text-[9px] text-muted-foreground shrink-0">
                {ind.measurement_dtype}
              </span>
            </div>
          ))}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-muted-foreground !w-2 !h-2"
      />
    </div>
  );
}

export const EffectNode = memo(EffectNodeInner);
