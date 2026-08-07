import type { CSSProperties } from "react";
import { DAG_COLORS } from "./palette";

export type DagFlowDirection = "horizontal" | "vertical";

interface DagDirectionToggleProps {
  direction: DagFlowDirection;
  onDirectionChange: (direction: DagFlowDirection) => void;
}

export function DagDirectionToggle({ direction, onDirectionChange }: DagDirectionToggleProps) {
  return (
    <>
      <span style={LABEL}>Flow</span>
      <div style={SEGMENTED_CONTROL}>
        {(["horizontal", "vertical"] as const).map((option) => (
          <button
            key={option}
            type="button"
            aria-pressed={direction === option}
            onClick={() => onDirectionChange(option)}
            style={segmentButton(direction === option)}
          >
            {option === "horizontal" ? "→ left to right" : "↓ top to bottom"}
          </button>
        ))}
      </div>
    </>
  );
}

const LABEL: CSSProperties = {
  fontSize: 11,
  letterSpacing: ".04em",
  textTransform: "uppercase",
  color: DAG_COLORS.muted,
};

const SEGMENTED_CONTROL: CSSProperties = {
  display: "inline-flex",
  background: "#eef0f3",
  borderRadius: 10,
  padding: 3,
};

const segmentButton = (active: boolean): CSSProperties => ({
  border: 0,
  background: active ? "#fff" : "transparent",
  padding: "7px 12px",
  borderRadius: 8,
  fontSize: 13,
  color: active ? DAG_COLORS.ink : "#4a4f57",
  cursor: "pointer",
  fontWeight: active ? 600 : 400,
  boxShadow: active ? "0 1px 2px rgba(0,0,0,.08)" : undefined,
});
