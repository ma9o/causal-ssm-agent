"use client";

import { ticks } from "d3-array";
import { useState } from "react";
import { clamp01, DAG_COLORS, signColor } from "../core/palette";

const { positive: TEAL, negative: RED, muted: MUTED, slate: SLATE, intervention: BLUE, ink: INK } =
  DAG_COLORS;

export interface NodeIntervention {
  day: number;
  value: number;
}

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
  /** Day axis (0..60). */
  days: number[];
  reference: number[];
  action: number[];
  realized: number[] | null;
  timeIndex: number;
  intervention: NodeIntervention | null;
  /** do() control state. */
  interactive?: boolean;
  otherActive?: boolean;
  onSetDo?: (value: number) => void;
  onRemoveDo?: () => void;
}

/**
 * One construct's node card: its counterfactual trajectory + credible bands +
 * realized data + playhead, with an in-card do() control. Exact port of the
 * playground's `drawCard` + `drawChart` (252×152).
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
  realized,
  timeIndex,
  intervention,
  interactive,
  otherActive,
  onSetDo,
  onRemoveDo,
}: TrajectoryCardProps) {
  const [hoverDay, setHoverDay] = useState<number | null>(null);

  const n = days.length;
  const day = Math.max(0, Math.min(n - 1, timeIndex));
  const latent = kind === "endo";
  const iv = isPrev ? null : intervention;
  const base = reference[0] ?? 0;

  // bands (heuristic forecast intervals — matches the playground's rebuildTraj)
  const refLo: number[] = [];
  const refHi: number[] = [];
  const lo: number[] = [];
  const hi: number[] = [];
  let moved = false;
  for (let t = 0; t < n; t++) {
    if (latent && Math.abs((action[t] ?? 0) - (reference[t] ?? 0)) > 0.003) moved = true;
    const refHalf = latent ? 0.02 + 0.0009 * t : 0;
    let cfHalf = latent ? 0.025 + 0.0016 * t : 0;
    if (iv && latent) cfHalf = Math.min(cfHalf, 0.004 + 0.005 * Math.abs(t - iv.day));
    refLo.push((reference[t] ?? 0) - refHalf);
    refHi.push((reference[t] ?? 0) + refHalf);
    lo.push((action[t] ?? 0) - cfHalf);
    hi.push((action[t] ?? 0) + cfHalf);
  }

  const x0 = 36;
  const x1 = w - 14;
  const y0 = h - 14;
  const y1 = 54;

  const vals: number[] = [];
  for (let t = 0; t < n; t++) {
    vals.push(reference[t] ?? 0);
    if (latent) {
      vals.push(refLo[t], refHi[t]);
      if (moved) vals.push(lo[t], hi[t], action[t] ?? 0);
    } else {
      vals.push(action[t] ?? 0);
    }
  }
  if (latent && realized) vals.push(...realized);
  let mn = Math.min(...vals);
  let mx = Math.max(...vals);
  const pad = Math.max((mx - mn) * 0.08, 0.02);
  mn -= pad;
  mx += pad;

  const sx = (t: number) => x0 + (t / 60) * (x1 - x0);
  const sy = (v: number) => y0 - ((v - mn) / (mx - mn)) * (y0 - y1);

  const line = (series: number[]) =>
    series.map((v, t) => `${t ? "L" : "M"}${sx(days[t]).toFixed(1)},${sy(v).toFixed(1)}`).join("");
  const area = (loS: number[], hiS: number[]) => {
    const up = hiS.map((v, t) => `${t ? "L" : "M"}${sx(days[t]).toFixed(1)},${sy(v).toFixed(1)}`).join("");
    const dn = loS
      .map((v, t) => ({ v, t }))
      .reverse()
      .map(({ v, t }) => `L${sx(days[t]).toFixed(1)},${sy(v).toFixed(1)}`)
      .join("");
    return `${up}${dn}Z`;
  };

  const pkey = latent && !moved ? reference : action;
  const atT = iv != null && day === iv.day;
  const border = isPrev ? "#d8dce2" : atT ? BLUE : signColor((action[day] ?? 0) - base);
  const strokeWidth = atT ? 2.6 : isTarget ? 2.0 : 1.4;

  const yTicks = ticks(mn, mx, 3);

  // hover crosshair readout
  const hov = hoverDay != null ? Math.max(0, Math.min(n - 1, hoverDay)) : null;

  return (
    <g opacity={isPrev ? 0.42 : 1}>
      <rect width={w} height={h} rx={11} fill="#fff" stroke={border} strokeWidth={strokeWidth} />
      <text x={14} y={24} fontSize={13} fontWeight={650} fill={INK}>
        {(isTarget ? "★ " : "") + name.replace(/_/g, " ") + (isPrev ? "  · t−1" : "")}
      </text>
      <text x={14} y={40} fontSize={9.5} fill={MUTED}>
        {`${kind} · ${vary}${kind === "exo" ? " · held" : ""}`}
      </text>

      {!isPrev && interactive ? (
        <foreignObject x={w - 120} y={28} width={108} height={22}>
          <div data-dag-interactive style={{ display: "flex", justifyContent: "flex-end" }}>
            {iv ? (
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
                {`do ${iv.value.toFixed(2)} @d${iv.day}`}
                <button
                  type="button"
                  onClick={onRemoveDo}
                  title="remove intervention"
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
              </div>
            ) : (
              <DoControl
                defaultValue={clamp01(action[day] ?? 0).toFixed(2)}
                disabled={!!otherActive}
                onSet={(v) => onSetDo?.(v)}
              />
            )}
          </div>
        </foreignObject>
      ) : null}

      {/* y-axis guides + labels */}
      {yTicks.map((tv) => (
        <g key={tv}>
          <line x1={x0} x2={x1} y1={sy(tv).toFixed(1)} y2={sy(tv).toFixed(1)} stroke="#f1f3f6" strokeWidth={1} />
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
      {[7, 30].map((gd) => (
        <line key={gd} x1={sx(gd)} x2={sx(gd)} y1={y1} y2={y0} stroke="#eef1f4" strokeWidth={1} />
      ))}

      {latent ? <path d={area(refLo, refHi)} fill={MUTED} fillOpacity={0.16} stroke="none" /> : null}
      {latent && moved ? <path d={area(lo, hi)} fill={TEAL} fillOpacity={0.16} stroke="none" /> : null}

      {latent && realized
        ? realized.slice(0, day + 1).map((v, t) => (
            <circle key={t} cx={sx(days[t]).toFixed(1)} cy={sy(v).toFixed(1)} r={1.3} fill={DAG_COLORS.realized} fillOpacity={0.75} />
          ))
        : null}

      {!latent ? <path d={line(action)} fill="none" stroke={MUTED} strokeWidth={2} /> : null}
      {latent && moved ? (
        <>
          <path d={line(reference)} fill="none" stroke={MUTED} strokeWidth={1.4} strokeDasharray="3,3" strokeOpacity={0.9} />
          <path d={line(action)} fill="none" stroke={TEAL} strokeWidth={2.4} />
        </>
      ) : null}
      {latent && !moved ? <path d={line(reference)} fill="none" stroke={SLATE} strokeWidth={2.2} /> : null}

      {iv ? (
        <>
          <line x1={sx(iv.day)} x2={sx(iv.day)} y1={y1} y2={y0} stroke={BLUE} strokeWidth={1} strokeDasharray="3,2" strokeOpacity={0.7} />
          <circle cx={sx(iv.day)} cy={sy(iv.value)} r={3.5} fill={BLUE} stroke="#fff" strokeWidth={1} />
          <text x={sx(iv.day)} y={y0 + 11} textAnchor="middle" fontSize={8} fontFamily="ui-monospace, monospace" fill={BLUE}>
            {`do @d${iv.day}`}
          </text>
        </>
      ) : null}

      {/* playhead */}
      <line x1={sx(days[day])} x2={sx(days[day])} y1={y1 - 2} y2={y0 + 2} stroke={RED} strokeWidth={1.2} />
      <circle cx={sx(days[day])} cy={sy(pkey[day] ?? 0)} r={3} fill={RED} />
      <text x={sx(days[day])} y={y1 - 5} textAnchor="middle" fontSize={10} fontFamily="ui-monospace, monospace" fill={RED}>
        {(pkey[day] ?? 0).toFixed(2)}
      </text>

      {/* hover-to-inspect */}
      {hov != null ? (
        <g pointerEvents="none">
          <line x1={sx(days[hov])} x2={sx(days[hov])} y1={y1} y2={y0} stroke={INK} strokeOpacity={0.35} strokeWidth={1} strokeDasharray="2,2" />
          {latent && moved ? (
            <circle cx={sx(days[hov])} cy={sy(reference[hov] ?? 0)} r={2.4} fill={MUTED} stroke="#fff" strokeWidth={1} />
          ) : null}
          <circle cx={sx(days[hov])} cy={sy(pkey[hov] ?? 0)} r={2.8} fill={moved ? TEAL : latent ? SLATE : MUTED} stroke="#fff" strokeWidth={1} />
          {(() => {
            const xx = sx(days[hov]);
            const right = xx > (x0 + x1) / 2;
            const dd = (action[hov] ?? 0) - (reference[hov] ?? 0);
            const txt = moved
              ? `d${hov} · ${(action[hov] ?? 0).toFixed(2)} · Δ${(dd >= 0 ? "+" : "") + dd.toFixed(2)}`
              : `d${hov} · ${(pkey[hov] ?? 0).toFixed(2)}`;
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
          const px = ((e.clientX - rect.left) / rect.width) * (x1 - x0);
          setHoverDay(Math.max(0, Math.min(60, Math.round((px / (x1 - x0)) * 60))));
        }}
        onMouseLeave={() => setHoverDay(null)}
      />
    </g>
  );
}

function DoControl({
  defaultValue,
  disabled,
  onSet,
}: {
  defaultValue: string;
  disabled?: boolean;
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
        min="0"
        max="1"
        value={text}
        disabled={disabled}
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
          opacity: disabled ? 0.4 : 1,
        }}
      />
      <button
        type="button"
        onClick={submit}
        disabled={disabled}
        style={{
          border: `1px solid ${disabled ? DAG_COLORS.line2 : BLUE}`,
          background: disabled ? "#cfd4db" : BLUE,
          color: "#fff",
          borderRadius: 5,
          cursor: disabled ? "not-allowed" : "pointer",
          fontSize: 10,
          fontWeight: 600,
          padding: "2px 7px",
          opacity: disabled ? 0.35 : 1,
        }}
      >
        set
      </button>
    </div>
  );
}
