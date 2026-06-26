"use client";

import { clamp01, DAG_COLORS } from "../core/palette";
import { CARD_H, CARD_W, IGAP, ISTACK_TOP, MINI_H } from "./build-cone-graph";
import type { IndicatorSeries } from "./contract-extension";

const { positive: TEAL, muted: MUTED, negative: RED, realized: DOT } = DAG_COLORS;
const MINI_W = CARD_W - 12;

/**
 * The manifest layer: a stack of measurement-channel mini-cards dropped below a
 * construct card when Indicators is on. Each contrasts the data we observed & fit
 * (gray points) with the counterfactual posterior-predictive under do() (teal).
 * Exact port of `drawIndicatorStack` + `drawIndicatorCard`.
 */
export function IndicatorStack({
  indicators,
  days,
  timeIndex,
  moved,
}: {
  indicators: IndicatorSeries[];
  days: number[];
  timeIndex: number;
  moved: boolean;
}) {
  if (indicators.length === 0) return null;
  return (
    <>
      <text x={6} y={CARD_H + 11} fontSize={8} fontWeight={600} fill={MUTED}>
        {`indicators · observed ● vs counterfactual ▔ (${indicators.length})`}
      </text>
      {indicators.map((ind, i) => (
        <g key={ind.id} transform={`translate(6,${CARD_H + ISTACK_TOP + i * (MINI_H + IGAP)})`}>
          <IndicatorMiniCard ind={ind} days={days} timeIndex={timeIndex} moved={moved} />
        </g>
      ))}
    </>
  );
}

function IndicatorMiniCard({
  ind,
  days,
  timeIndex,
  moved,
}: {
  ind: IndicatorSeries;
  days: number[];
  timeIndex: number;
  moved: boolean;
}) {
  const W = MINI_W;
  const H = MINI_H;
  const binary = ind.type === "binary";
  const n = days.length;
  const day = Math.max(0, Math.min(n - 1, timeIndex));

  const x0 = 6;
  const x1 = W - 6;
  const y0 = H - 5;
  const y1 = 15;
  const X = (t: number) => x0 + (t / 60) * (x1 - x0);

  let lo: number;
  let hi: number;
  if (binary) {
    lo = -0.1;
    hi = 1.1;
  } else {
    const vs = [...ind.observed.map((o) => o.v), ...ind.ref, ...(moved ? ind.cf : [])];
    lo = Math.min(...vs);
    hi = Math.max(...vs);
    const pad = (hi - lo) * 0.18 || 0.05;
    lo -= pad;
    hi += pad;
  }
  const Y = (v: number) => y0 - ((v - lo) / (hi - lo)) * (y0 - y1);
  const lineOf = (series: number[]) =>
    series.map((m, t) => `${t ? "L" : "M"}${X(days[t]).toFixed(1)},${Y(m).toFixed(1)}`).join("");

  const ribbon = (() => {
    if (!(moved && !binary)) return null;
    const up = ind.cf.map((m, t) => `${t ? "L" : "M"}${X(days[t]).toFixed(1)},${Y(clamp01(m + 1.96 * ind.sd)).toFixed(1)}`).join("");
    const dn = ind.cf
      .map((m, t) => ({ m, t }))
      .reverse()
      .map(({ m, t }) => `L${X(days[t]).toFixed(1)},${Y(clamp01(m - 1.96 * ind.sd)).toFixed(1)}`)
      .join("");
    return `${up}${dn}Z`;
  })();

  const cur = (moved ? ind.cf : ind.ref)[day] ?? 0;

  return (
    <>
      <rect width={W} height={H} rx={6} fill="#fff" stroke="#e6e9ee" strokeWidth={1} />
      <text x={6} y={10} fontSize={7.5} fontWeight={600} fill="#3a3f47">
        {ind.id.replace(/_/g, " ")}
      </text>
      <text x={W - 5} y={10} textAnchor="end" fontSize={6.5} fontFamily="ui-monospace, monospace" fill={MUTED}>
        {binary ? "0/1" : "cont"}
      </text>

      {ribbon ? <path d={ribbon} fill={TEAL} fillOpacity={0.14} stroke="none" /> : null}
      <path d={lineOf(ind.ref)} fill="none" stroke={MUTED} strokeWidth={1.2} strokeOpacity={0.85} strokeDasharray="3,2" />
      {moved ? <path d={lineOf(ind.cf)} fill="none" stroke={TEAL} strokeWidth={1.6} /> : null}

      {ind.observed
        .filter((o) => o.t <= days[day])
        .map((o) => (
          <circle key={o.t} cx={X(o.t).toFixed(1)} cy={Y(o.v).toFixed(1)} r={1.5} fill={DOT} fillOpacity={0.85} />
        ))}

      <line x1={X(days[day])} x2={X(days[day])} y1={y1} y2={y0} stroke={RED} strokeWidth={1} strokeOpacity={0.8} />
      <circle cx={X(days[day])} cy={Y(cur)} r={2} fill={RED} />
    </>
  );
}
