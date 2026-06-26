"use client";

import { DAG_COLORS } from "../core/palette";

interface DriftGlyphProps {
  width: number;
  height: number;
  /** Contribution as a function of driver level c(s), s = i/N over [0,1] (panel A). */
  transfer: number[];
  /** Signed drift contribution over the effect-trajectory days (panel B). */
  contribution: number[];
  /** Driver level over days — the panel-A operating point's x. */
  driverLevel: number[];
  timeIndex: number;
  /** Sign color of the contribution at the current day. */
  color: string;
  /** Short tag: lin / Hill / × / self. */
  label: string;
  /** Panel-A x caption: "vs cause" or "vs level". */
  xlabel: string;
  highlighted?: boolean;
}

/**
 * The two-panel drift glyph the playground puts on every edge and self-effect:
 * LEFT = the contribution as a function of the driving level (transfer shape),
 * RIGHT = that contribution over time. Both share one auto-ranged signed y-axis
 * with a zero line; a dark operating dot marks the present day on both. Exact
 * port of the playground's `drawEffectGlyph` (86×36).
 */
export function DriftGlyph({
  width,
  height,
  transfer,
  contribution,
  driverLevel,
  timeIndex,
  color,
  label,
  xlabel,
  highlighted,
}: DriftGlyphProps) {
  const W = width;
  const H = height;
  const n = contribution.length;
  if (n === 0 || transfer.length === 0) return null;
  const day = Math.max(0, Math.min(n - 1, timeIndex));
  const N = transfer.length - 1;

  const all = [...transfer, ...contribution];
  const yTop = 14;
  const yBot = H - 7;
  let lo = Math.min(0, ...all);
  let hi = Math.max(0, ...all);
  const pad = (hi - lo) * 0.14 || 1;
  lo -= pad;
  hi += pad;
  const Yc = (v: number) => yBot - ((v - lo) / (hi - lo)) * (yBot - yTop);
  const z = Yc(0);

  const xA0 = 6;
  const xA1 = 40;
  const XA = (s: number) => xA0 + Math.max(0, Math.min(1, s)) * (xA1 - xA0);
  const xB0 = 46;
  const xB1 = 80;
  const XB = (t: number) => xB0 + (t / 60) * (xB1 - xB0);

  const s0 = driverLevel[day] ?? 0;
  const cDay = contribution[day] ?? 0;

  const aPath = transfer
    .map((v, i) => `${i ? "L" : "M"}${XA(i / N).toFixed(1)},${Yc(v).toFixed(1)}`)
    .join("");
  const bSeg = (upto: number) =>
    contribution
      .slice(0, upto)
      .map((v, t) => `${t ? "L" : "M"}${XB(t).toFixed(1)},${Yc(v).toFixed(1)}`)
      .join("");

  return (
    <g>
      <rect
        width={W}
        height={H}
        rx={7}
        fill="#fff"
        fillOpacity={0.97}
        stroke={color}
        strokeOpacity={highlighted ? 1 : 0.5}
        strokeWidth={highlighted ? 2 : 1}
      />
      <text x={6} y={9.5} fontSize={7} fontFamily="ui-monospace, monospace" fill={DAG_COLORS.muted}>
        {label}
      </text>

      {/* zero lines */}
      <line x1={xA0} x2={xA1} y1={z} y2={z} stroke="#e7ebef" strokeWidth={1} />
      <line x1={xB0} x2={xB1} y1={z} y2={z} stroke="#e7ebef" strokeWidth={1} />

      {/* panel A — contribution vs driving level */}
      <path d={aPath} fill="none" stroke={color} strokeWidth={1.6} strokeLinecap="round" />
      <line
        x1={XA(s0)}
        x2={XA(s0)}
        y1={z}
        y2={Yc(cDay)}
        stroke={DAG_COLORS.ink}
        strokeOpacity={0.35}
        strokeWidth={1}
      />
      <circle cx={XA(s0).toFixed(1)} cy={Yc(cDay).toFixed(1)} r={2.2} fill={DAG_COLORS.ink} />

      {/* divider */}
      <line x1={43} x2={43} y1={13} y2={H - 4} stroke="#eef1f4" strokeWidth={1} />

      {/* panel B — contribution over time */}
      <path d={bSeg(n)} fill="none" stroke={color} strokeWidth={1} strokeOpacity={0.25} strokeLinecap="round" />
      <path d={bSeg(day + 1)} fill="none" stroke={color} strokeWidth={1.6} strokeOpacity={1} strokeLinecap="round" />
      <circle cx={XB(day).toFixed(1)} cy={Yc(cDay).toFixed(1)} r={2.2} fill={DAG_COLORS.ink} />

      <text x={(xA0 + xA1) / 2} y={H - 1} textAnchor="middle" fontSize={5.5} fill={DAG_COLORS.muted}>
        {xlabel}
      </text>
      <text x={(xB0 + xB1) / 2} y={H - 1} textAnchor="middle" fontSize={5.5} fill={DAG_COLORS.muted}>
        over time
      </text>
    </g>
  );
}
