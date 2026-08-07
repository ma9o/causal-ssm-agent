import type { CSSProperties } from "react";
import { DAG_COLORS } from "./palette";

interface DagZoomControlsProps {
  zoom: number;
  onZoomChange: (zoom: number) => void;
}

export function DagZoomControls({ zoom, onZoomChange }: DagZoomControlsProps) {
  return (
    <>
      <span style={LABEL}>Zoom</span>
      <div style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
        <button
          type="button"
          onClick={() => onZoomChange(zoom / 1.2)}
          style={ZOOM_BUTTON}
          title="zoom out"
        >
          −
        </button>
        <span style={ZOOM_VALUE}>{Math.round(zoom * 100)}%</span>
        <button
          type="button"
          onClick={() => onZoomChange(zoom * 1.2)}
          style={ZOOM_BUTTON}
          title="zoom in"
        >
          +
        </button>
        <button
          type="button"
          onClick={() => onZoomChange(1)}
          style={ZOOM_BUTTON}
          title="reset zoom"
        >
          ⤢
        </button>
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

const ZOOM_VALUE: CSSProperties = {
  fontVariantNumeric: "tabular-nums",
  fontSize: 12,
  color: "#4a4f57",
  minWidth: 40,
  textAlign: "center",
};

const ZOOM_BUTTON: CSSProperties = {
  width: 26,
  height: 26,
  border: `1px solid ${DAG_COLORS.line2}`,
  background: "#fff",
  borderRadius: 7,
  cursor: "pointer",
  fontSize: 14,
  lineHeight: 1,
  display: "grid",
  placeItems: "center",
  color: "#4a4f57",
};
