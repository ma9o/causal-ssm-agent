"use client";

import type { LatentClampInput } from "@nof1-causal-lab/api-types";
import { ticks } from "d3-array";
import { useState } from "react";
import { DAG_COLORS, signColor } from "../core/palette";
import { formatClampValue } from "../intervention-dag-semantics";
import type { EdgePosterior } from "../intervention-dag-types";
import type { ConstructStatus } from "../structure-dag";

const {
  positive: TEAL,
  negative: RED,
  muted: MUTED,
  slate: SLATE,
  intervention: BLUE,
  ink: INK,
} = DAG_COLORS;

interface TrajectoryCardProps {
  width: number;
  height: number;
  name: string;
  /** "endo" | "exo" */
  kind: string;
  /** "varying" | "invariant" */
  vary: string;
  isTarget: boolean;
  /** t−1 unrolled ghost card. */
  isPrev: boolean;
  /** Day axis supplied by the simulation result. */
  days: number[];
  reference: number[];
  action: number[];
  timeIndex: number;
  interventions: LatentClampInput[];
  /** Backend disposition/identification status for retained theory context. */
  status?: ConstructStatus;
  /** Compiled as an observed transition input rather than a latent state. */
  knownInput?: boolean;
  /** Fitted baseline daily-persistence posterior for an executable latent state. */
  persistence?: EdgePosterior;
  /** do() control state. */
  interactive?: boolean;
  onSetDo?: (value: number) => void;
  onRemoveDo?: () => void;
}

/**
 * One construct's node card. The chart renders only trajectories materialized
 * by the backend simulation result; constructs outside that projection remain
 * visible as scientific context without an invented series.
 */
export function TrajectoryCard({
  width: w,
  height: h,
  name,
  kind,
  vary,
  isTarget,
  isPrev,
  days,
  reference,
  action,
  timeIndex,
  interventions,
  status,
  knownInput = false,
  persistence,
  interactive,
  onSetDo,
  onRemoveDo,
}: TrajectoryCardProps) {
  const [hoverDay, setHoverDay] = useState<number | null>(null);
  const contextOnly = status === "marginalized";

  if (isPrev) {
    return (
      <g opacity={0.42}>
        <rect
          width={w}
          height={h}
          rx={11}
          fill={contextOnly ? "#f1f3f5" : "#fff"}
          stroke={DAG_COLORS.line2}
          strokeWidth={1.4}
          strokeDasharray="5,4"
        />
        <text x={14} y={24} fontSize={13} fontWeight={650} fill={INK}>
          {`${name.replace(/_/g, " ")} · t−1`}
        </text>
        <text x={14} y={43} fontSize={9.5} fill={MUTED}>
          {persistence
            ? `fitted daily persistence · ρ ${persistence.mean.toFixed(2)}`
            : "previous-time construct"}
        </text>
      </g>
    );
  }

  if (reference.length > 0 && reference.length !== days.length) {
    throw new Error(
      `Invalid simulation visualization for ${name}: reference trajectory is not aligned to effect_trajectory`,
    );
  }
  if (action.length > 0 && action.length !== days.length) {
    throw new Error(
      `Invalid simulation visualization for ${name}: action trajectory is not aligned to effect_trajectory`,
    );
  }
  if (action.length > 0 && reference.length === 0) {
    throw new Error(
      `Invalid simulation visualization for ${name}: action trajectory has no reference trajectory`,
    );
  }

  const n = reference.length > 0 ? days.length : 0;
  const hasReference = n > 0;
  const hasAction = n > 0 && action.length === n;
  const day = Math.max(0, Math.min(n - 1, timeIndex));
  const latent = kind === "endo";
  const nodeInterventions = interventions;

  const actionDiffers =
    hasAction &&
    action.slice(0, n).some((value, index) => Math.abs(value - reference[index]) > 1e-12);

  const x0 = 36;
  const x1 = w - 14;
  const y0 = h - 14;
  const y1 = 54;

  const vals = hasReference
    ? [...reference.slice(0, n), ...(hasAction ? action.slice(0, n) : [])]
    : [0, 1];
  let mn = Math.min(...vals);
  let mx = Math.max(...vals);
  const pad = Math.max((mx - mn) * 0.08, 0.02);
  mn -= pad;
  mx += pad;

  const firstDay = hasReference ? days[0] : 0;
  const lastDay = hasReference ? days[n - 1] : 1;
  const dayRange = lastDay - firstDay || 1;
  const sx = (t: number) => x0 + ((t - firstDay) / dayRange) * (x1 - x0);
  const sy = (v: number) => y0 - ((v - mn) / (mx - mn)) * (y0 - y1);

  const line = (series: number[]) =>
    series
      .slice(0, n)
      .map((v, t) => `${t ? "L" : "M"}${sx(days[t]).toFixed(1)},${sy(v).toFixed(1)}`)
      .join("");

  const pkey = hasAction ? action : reference;
  const currentDay = hasReference ? days[day] : null;
  const currentReference = hasReference ? reference[day] : null;
  const currentAction = hasAction ? action[day] : null;
  const atT =
    currentDay != null && nodeInterventions.some((clamp) => currentDay === clamp.from_day);
  const border = contextOnly
    ? DAG_COLORS.line2
    : isPrev
      ? "#d8dce2"
      : atT
        ? BLUE
        : signColor(
            currentAction != null && currentReference != null
              ? currentAction - currentReference
              : 0,
          );
  const strokeWidth = atT ? 2.6 : isTarget ? 2.0 : 1.4;

  const yTicks = ticks(mn, mx, 3);

  // hover crosshair readout
  const hov = hoverDay != null ? Math.max(0, Math.min(n - 1, hoverDay)) : null;

  return (
    <g opacity={contextOnly ? 0.46 : 1}>
      <rect
        width={w}
        height={h}
        rx={11}
        fill={contextOnly ? "#f1f3f5" : "#fff"}
        stroke={border}
        strokeWidth={strokeWidth}
        strokeDasharray={contextOnly ? "5,4" : undefined}
      />
      <text x={14} y={24} fontSize={13} fontWeight={650} fill={INK}>
        {(isTarget ? "★ " : "") + name.replace(/_/g, " ")}
      </text>
      <text x={14} y={40} fontSize={9.5} fill={MUTED}>
        {`${kind === "endo" ? "theory endo" : "theory exo"} · ${vary}${knownInput ? " · known input" : kind === "exo" ? " · held" : ""}${status === "marginalized" ? " · marginalized" : ""}`}
      </text>
      {persistence ? (
        <text
          x={w - 14}
          y={50}
          textAnchor="end"
          fontSize={8}
          fontFamily="ui-monospace, monospace"
          fill={MUTED}
        >
          {`ρ ${persistence.mean.toFixed(2)} [${persistence.ci_lower.toFixed(2)}, ${persistence.ci_upper.toFixed(2)}]`}
        </text>
      ) : null}

      {interactive && !contextOnly && currentReference != null ? (
        <foreignObject x={w - 120} y={28} width={108} height={22}>
          <div data-dag-interactive style={{ display: "flex", justifyContent: "flex-end" }}>
            {nodeInterventions.length > 0 && onRemoveDo ? (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 5,
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 11,
                  fontWeight: 700,
                  color: BLUE,
                }}
              >
                {nodeInterventions.length === 1 && nodeInterventions[0]
                  ? `${formatClampValue(nodeInterventions[0])} @d${nodeInterventions[0].from_day}`
                  : `${nodeInterventions.length} clamps`}
                {onRemoveDo ? (
                  <button
                    type="button"
                    onClick={onRemoveDo}
                    title="reset to the selected scenario"
                    style={{
                      border: `1px solid ${RED}`,
                      background: "#fff",
                      color: RED,
                      borderRadius: 5,
                      cursor: "pointer",
                      fontSize: 11,
                      fontWeight: 700,
                      lineHeight: 1,
                      padding: "1px 5px",
                    }}
                  >
                    ✕
                  </button>
                ) : null}
              </div>
            ) : (
              <DoControl
                defaultValue={(currentAction ?? currentReference).toFixed(2)}
                onSet={(v) => onSetDo?.(v)}
              />
            )}
          </div>
        </foreignObject>
      ) : null}

      {hasReference ? (
        <>
          {/* y-axis guides + labels */}
          {yTicks.map((tv) => (
            <g key={tv}>
              <line
                x1={x0}
                x2={x1}
                y1={sy(tv).toFixed(1)}
                y2={sy(tv).toFixed(1)}
                stroke="#f1f3f6"
                strokeWidth={1}
              />
              <text
                x={x0 - 5}
                y={(sy(tv) + 2.6).toFixed(1)}
                textAnchor="end"
                fontSize={8}
                fontFamily="ui-monospace, monospace"
                fill={MUTED}
              >
                {tv.toFixed(2)}
              </text>
            </g>
          ))}
          <line x1={x0} x2={x0} y1={y1} y2={y0} stroke={DAG_COLORS.line} strokeWidth={1} />
          {days
            .filter((value) => value > firstDay && value < lastDay)
            .filter(
              (_, index, values) =>
                index === Math.floor(values.length / 3) ||
                index === Math.floor((values.length * 2) / 3),
            )
            .map((gd) => (
              <line
                key={gd}
                x1={sx(gd)}
                x2={sx(gd)}
                y1={y1}
                y2={y0}
                stroke="#eef1f4"
                strokeWidth={1}
              />
            ))}

          {hasAction ? (
            <>
              <path
                d={line(reference)}
                fill="none"
                stroke={MUTED}
                strokeWidth={actionDiffers ? 1.4 : 3.4}
                strokeDasharray={actionDiffers ? "3,3" : undefined}
                strokeOpacity={0.9}
              />
              <path
                d={line(action)}
                fill="none"
                stroke={TEAL}
                strokeWidth={actionDiffers ? 2.4 : 1.5}
                strokeDasharray={actionDiffers ? undefined : "2,2"}
              />
            </>
          ) : null}
          {!hasAction ? (
            <path d={line(reference)} fill="none" stroke={SLATE} strokeWidth={2.2} />
          ) : null}

          {nodeInterventions.map((clamp, clampIndex) => (
            <g key={`${clamp.variable}-${clamp.from_day}-${clampIndex}`}>
              <line
                x1={sx(clamp.from_day)}
                x2={sx(clamp.from_day)}
                y1={y1}
                y2={y0}
                stroke={BLUE}
                strokeWidth={1}
                strokeDasharray="3,2"
                strokeOpacity={0.7}
              />
              {clamp.mode === "set" && clamp.value != null ? (
                <circle
                  cx={sx(clamp.from_day)}
                  cy={sy(clamp.value)}
                  r={3.5}
                  fill={BLUE}
                  stroke="#fff"
                  strokeWidth={1}
                />
              ) : null}
              <text
                x={sx(clamp.from_day)}
                y={y0 + 11 + (clampIndex % 2) * 9}
                textAnchor="middle"
                fontSize={8}
                fontFamily="ui-monospace, monospace"
                fill={BLUE}
              >
                {`do @d${clamp.from_day}`}
              </text>
            </g>
          ))}

          {/* playhead */}
          <line
            x1={sx(days[day])}
            x2={sx(days[day])}
            y1={y1 - 2}
            y2={y0 + 2}
            stroke={RED}
            strokeWidth={1.2}
          />
          <circle cx={sx(days[day])} cy={sy(pkey[day])} r={3} fill={RED} />
          <text
            x={sx(days[day])}
            y={y1 - 5}
            textAnchor="middle"
            fontSize={10}
            fontFamily="ui-monospace, monospace"
            fill={RED}
          >
            {pkey[day].toFixed(2)}
          </text>

          {/* hover-to-inspect */}
          {hov != null ? (
            <g pointerEvents="none">
              <line
                x1={sx(days[hov])}
                x2={sx(days[hov])}
                y1={y1}
                y2={y0}
                stroke={INK}
                strokeOpacity={0.35}
                strokeWidth={1}
                strokeDasharray="2,2"
              />
              {latent && hasAction ? (
                <circle
                  cx={sx(days[hov])}
                  cy={sy(reference[hov])}
                  r={2.4}
                  fill={MUTED}
                  stroke="#fff"
                  strokeWidth={1}
                />
              ) : null}
              <circle
                cx={sx(days[hov])}
                cy={sy(pkey[hov])}
                r={2.8}
                fill={hasAction ? TEAL : latent ? SLATE : MUTED}
                stroke="#fff"
                strokeWidth={1}
              />
              {(() => {
                const xx = sx(days[hov]);
                const right = xx > (x0 + x1) / 2;
                const dd = hasAction ? action[hov] - reference[hov] : 0;
                const txt = hasAction
                  ? `d${days[hov]} · ${action[hov].toFixed(2)} · Δ${(dd >= 0 ? "+" : "") + dd.toFixed(2)}`
                  : `d${days[hov]} · ${pkey[hov].toFixed(2)}`;
                return (
                  <text
                    x={(right ? xx - 5 : xx + 5).toFixed(1)}
                    y={y1 + 9}
                    textAnchor={right ? "end" : "start"}
                    fontSize={8.5}
                    fontFamily="ui-monospace, monospace"
                    fill={INK}
                    stroke="#fff"
                    strokeWidth={2.6}
                    style={{ paintOrder: "stroke" }}
                  >
                    {txt}
                  </text>
                );
              })()}
            </g>
          ) : null}

          {/* hover capture */}
          <rect
            x={x0}
            y={y1}
            width={x1 - x0}
            height={y0 - y1}
            fill="transparent"
            style={{ cursor: "crosshair" }}
            onMouseMove={(e) => {
              const rect = (e.currentTarget as SVGRectElement).getBoundingClientRect();
              const fraction = (e.clientX - rect.left) / rect.width;
              setHoverDay(Math.max(0, Math.min(n - 1, Math.round(fraction * (n - 1)))));
            }}
            onMouseLeave={() => setHoverDay(null)}
          />
        </>
      ) : (
        <g>
          <rect x={14} y={55} width={w - 28} height={h - 69} rx={7} fill="#f4f5f7" />
          <text x={w / 2} y={91} textAnchor="middle" fontSize={9.5} fill={MUTED}>
            {knownInput ? "observed transition input u(t)" : "not materialized in this simulation"}
          </text>
          <text x={w / 2} y={107} textAnchor="middle" fontSize={8} fill="#9aa0a8">
            {knownInput
              ? "not part of the latent trajectory projection"
              : "retained as scientific DAG context"}
          </text>
        </g>
      )}
    </g>
  );
}

function DoControl({
  defaultValue,
  onSet,
}: {
  defaultValue: string;
  onSet: (value: number) => void;
}) {
  const [text, setText] = useState(defaultValue);
  const submit = () => {
    const v = Number.parseFloat(text);
    if (Number.isFinite(v)) onSet(v);
  };
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4, justifyContent: "flex-end" }}>
      <input
        type="number"
        step="0.01"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            submit();
          }
        }}
        style={{
          width: 42,
          fontFamily: "ui-monospace, monospace",
          fontSize: 11,
          padding: "2px 4px",
          border: `1px solid ${DAG_COLORS.line2}`,
          borderRadius: 5,
          textAlign: "right",
          color: "#26303f",
          background: "#fff",
        }}
      />
      <button
        type="button"
        onClick={submit}
        style={{
          border: `1px solid ${BLUE}`,
          background: BLUE,
          color: "#fff",
          borderRadius: 5,
          cursor: "pointer",
          fontSize: 10,
          fontWeight: 600,
          padding: "2px 7px",
        }}
      >
        set
      </button>
    </div>
  );
}
