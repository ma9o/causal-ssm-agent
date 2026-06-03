"use client";

import { cn } from "@/lib/utils";
import type { DagMetric } from "./intervention-dag-view-model";

function Segmented<T extends string>({
  value,
  onChange,
  options,
  disabled,
}: {
  value: T;
  onChange: (next: T) => void;
  options: { value: T; label: string }[];
  disabled?: boolean;
}) {
  return (
    <div className={cn("inline-flex rounded-md border p-0.5", disabled && "opacity-50")}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          disabled={disabled}
          onClick={() => onChange(option.value)}
          className={cn(
            "rounded px-2 py-0.5 text-[11px] font-medium transition-colors disabled:cursor-not-allowed",
            value === option.value
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/**
 * Presentational re-slice of a fixed simulation: which trajectory the DAG node
 * sparklines and the trajectory chart emphasise. (The latent↔manifest space
 * toggle is deferred — simulations don't carry per-indicator trajectories.)
 */
export function ViewControls({
  metric,
  onMetricChange,
  disabled = false,
}: {
  metric: DagMetric;
  onMetricChange: (metric: DagMetric) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] text-muted-foreground">Show</span>
        <Segmented
          value={metric}
          onChange={onMetricChange}
          disabled={disabled}
          options={[
            { value: "effect", label: "Effect Δ" },
            { value: "action", label: "Action path" },
            { value: "reference", label: "Reference" },
          ]}
        />
      </div>
    </div>
  );
}
