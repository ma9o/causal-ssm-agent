"use client";

import type { Stage4Point, Stage4SectionEdgeKind } from "@/lib/stage4-section-graph";
import type { EdgeProps } from "@xyflow/react";

interface Stage4SectionEdgeData {
  kind: Stage4SectionEdgeKind;
  points: Stage4Point[];
}

const ARROW_SIZE = 10;

function markerId(id: string, color: string): string {
  return `stage4-section-arrow-${id.replace(/[^a-z0-9]/gi, "")}-${color.replace(/[^a-z0-9]/gi, "")}`;
}

function buildPath(points: Stage4Point[]): string {
  if (points.length === 0) return "";
  const [first, ...rest] = points;
  return `M ${first.x} ${first.y} ${rest.map((point) => `L ${point.x} ${point.y}`).join(" ")}`;
}

export function Stage4SectionEdge({
  id,
  data,
  style,
  animated,
}: EdgeProps) {
  const edgeData = data as Stage4SectionEdgeData | undefined;
  const points = edgeData?.points ?? [];
  const path = buildPath(points);
  const stroke =
    typeof style?.stroke === "string"
      ? style.stroke
      : edgeData?.kind === "repair_transition"
        ? "var(--edge-lagged)"
        : "var(--edge-contemporary)";
  const strokeWidth = typeof style?.strokeWidth === "number" ? style.strokeWidth : 2;
  const opacity = typeof style?.opacity === "number" ? style.opacity : 1;
  const dashArray =
    typeof style?.strokeDasharray === "string"
      ? style.strokeDasharray
      : edgeData?.kind === "repair_transition"
        ? "6,4"
        : undefined;
  const marker = `url(#${markerId(id, stroke)})`;

  if (!path) return null;

  return (
    <g>
      <defs>
        <marker
          id={markerId(id, stroke)}
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
            fill={stroke}
            stroke={stroke}
            strokeWidth="1"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        </marker>
      </defs>
      <path d={path} fill="none" stroke="transparent" strokeWidth={20} />
      <path
        d={path}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeDasharray={animated ? dashArray ?? "8,6" : dashArray}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={opacity}
        markerEnd={marker}
      >
        {animated ? (
          <animate
            attributeName="stroke-dashoffset"
            from="14"
            to="0"
            dur="0.9s"
            repeatCount="indefinite"
          />
        ) : null}
      </path>
    </g>
  );
}
