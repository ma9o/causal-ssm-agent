"use client";

import type { Point } from "@/lib/utils/dag-graph-layout";
import { orthoPath } from "./ortho-path";

interface DagEdgeProps {
  /** Routed polyline from the layout: [start, ...bends, end]. */
  points: Point[];
  color: string;
  /** Stroke width in layout units. Highlighted edges render a touch thicker. */
  width?: number;
  dashed?: boolean;
  opacity?: number;
  /** Draw the arrowhead at the target end. */
  markerEnd?: boolean;
  highlighted?: boolean;
  onHoverChange?: (hovered: boolean) => void;
}

const ARROW_SIZE = 11;

/** Stable marker id per color, so an edge's arrowhead matches its stroke. */
function markerId(color: string): string {
  return `dag-arrow-${color.replace(/[^a-z0-9]/gi, "")}`;
}

/**
 * One routed causal edge: a softened-orthogonal line drawn directly from ELK's
 * routing, plus a fat transparent hit-path for hover. Color/width/dash are
 * supplied by the caller (structure status, or the simulation drift sign).
 */
export function DagEdge({
  points,
  color,
  width = 2,
  dashed,
  opacity = 0.95,
  markerEnd = true,
  highlighted,
  onHoverChange,
}: DagEdgeProps) {
  if (points.length < 2) return null;

  const d = orthoPath(points);
  const mid = markerId(color);
  const strokeWidth = highlighted ? width + 1.5 : width;

  return (
    <g>
      {markerEnd ? (
        <defs>
          <marker
            id={mid}
            markerWidth={ARROW_SIZE}
            markerHeight={ARROW_SIZE}
            viewBox="-5 -5 10 10"
            markerUnits="userSpaceOnUse"
            orient="auto-start-reverse"
            refX="0"
            refY="0"
          >
            <polyline
              points="-5,-4 0,0 -5,4 -5,-4"
              fill={color}
              stroke={color}
              strokeWidth="1"
              strokeLinejoin="round"
              strokeLinecap="round"
            />
          </marker>
        </defs>
      ) : null}

      {/* Fat invisible hover target. */}
      <path
        d={d}
        fill="none"
        stroke="transparent"
        strokeWidth={Math.max(16, strokeWidth + 12)}
        style={{ cursor: "pointer" }}
        onPointerEnter={() => onHoverChange?.(true)}
        onPointerLeave={() => onHoverChange?.(false)}
      />

      <path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeDasharray={dashed ? "6,4" : undefined}
        strokeOpacity={opacity}
        strokeLinecap="round"
        markerEnd={markerEnd ? `url(#${mid})` : undefined}
        style={{ transition: "stroke 200ms, stroke-width 150ms" }}
      />
    </g>
  );
}
