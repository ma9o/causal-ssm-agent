"use client";

import type { EdgeAnimState, EdgePosterior } from "./intervention-dag-types";
import {
  getBezierPath,
  getSmoothStepPath,
  type EdgeProps,
} from "@xyflow/react";
import { motion } from "motion/react";

interface WeightedEdgeData {
  cause: string;
  effect: string;
  description: string;
  lagged: boolean;
  posterior?: EdgePosterior;
  animState?: EdgeAnimState;
}

/** Map |mean effect| to stroke width (1–12 px) — quadratic scaling. */
function effectToWidth(mean: number): number {
  const abs = Math.abs(mean);
  return Math.max(1, Math.min(12, 1 + abs * abs * 26));
}

/** Positive → teal, negative → rose. */
function effectToColor(mean: number): string {
  return mean >= 0 ? "var(--color-teal-500)" : "var(--color-rose-500)";
}

/** Narrow CI → opaque, wide CI → translucent. */
function ciToOpacity(p: EdgePosterior): number {
  const width = p.ci_upper - p.ci_lower;
  return Math.max(0.35, Math.min(1, 1.1 - width * 0.6));
}

const TRANSITION = "all 500ms cubic-bezier(0.4, 0, 0.2, 1)";
const ARROW_SIZE = 10;

// Flow pulse parameters
const DASH = 8;
const GAP = 10;
const CYCLE = DASH + GAP;

/** Stable marker ID from color string (stripped of CSS var syntax). */
function markerId(color: string): string {
  return `weighted-arrow-${color.replace(/[^a-z0-9]/gi, "")}`;
}

export function WeightedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  data,
}: EdgeProps) {
  const d = data as unknown as WeightedEdgeData;
  const posterior = d?.posterior;
  const animState = d?.animState ?? "normal";
  const isLagged = d?.lagged;

  const pathParams = {
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  };
  const [edgePath] = isLagged
    ? getBezierPath(pathParams)
    : getSmoothStepPath({ ...pathParams, borderRadius: 8 });

  const strokeWidth = posterior ? effectToWidth(posterior.mean) : 2;
  const stroke = posterior
    ? effectToColor(posterior.mean)
    : isLagged
      ? "var(--edge-lagged)"
      : "var(--edge-contemporary)";
  const baseOpacity = posterior ? ciToOpacity(posterior) : 1;

  let opacity = baseOpacity;
  let dashArray: string | undefined = isLagged ? "6,4" : undefined;

  switch (animState) {
    case "cut":
      opacity = 0.1;
      dashArray = "3,14";
      break;
    case "flowing":
      opacity = Math.min(1, baseOpacity + 0.15);
      break;
    case "dimmed":
      opacity = 0.18;
      break;
  }

  const mid = markerId(stroke);
  const marker = `url(#${mid})`;

  const arrowDef = (
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
          fill={stroke}
          stroke={stroke}
          strokeWidth="1"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
      </marker>
    </defs>
  );

  // ── Flowing: solid pipe + animated white pulses ───────────────────
  if (animState === "flowing") {
    return (
      <g>
        {arrowDef}
        {/* Hit area */}
        <path d={edgePath} fill="none" stroke="transparent" strokeWidth={20} />
        {/* Outer glow halo */}
        <path
          d={edgePath}
          fill="none"
          stroke={stroke}
          strokeWidth={strokeWidth + 8}
          strokeLinecap="round"
          opacity={0.08}
          style={{ transition: TRANSITION }}
        />
        {/* Solid pipe base */}
        <path
          id={id}
          d={edgePath}
          fill="none"
          stroke={stroke}
          strokeWidth={strokeWidth + 1}
          strokeLinecap="round"
          opacity={opacity}
          markerEnd={marker}
          style={{ transition: TRANSITION }}
        />
        {/* Flow pulses — white dashes moving source → target */}
        <motion.path
          d={edgePath}
          fill="none"
          stroke="white"
          strokeWidth={Math.max(2.5, strokeWidth * 0.55)}
          strokeDasharray={`${DASH} ${GAP}`}
          strokeLinecap="round"
          opacity={0.6}
          initial={{ strokeDashoffset: CYCLE }}
          animate={{ strokeDashoffset: 0 }}
          transition={{
            duration: 0.8,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      </g>
    );
  }

  // ── Default: static weighted edge ─────────────────────────────────
  return (
    <g>
      {arrowDef}
      <path d={edgePath} fill="none" stroke="transparent" strokeWidth={20} />
      <path
        id={id}
        d={edgePath}
        fill="none"
        stroke={stroke}
        strokeWidth={strokeWidth}
        strokeDasharray={dashArray}
        strokeLinecap="round"
        opacity={opacity}
        markerEnd={marker}
        style={{ transition: TRANSITION }}
      />
    </g>
  );
}
