"use client";

import type { Indicator } from "@nof1-causal-lab/api-types";
import { DAG_COLORS } from "../core/palette";
import { CARD_H, CARD_W, IGAP, ISTACK_TOP, MINI_H } from "./build-cone-graph";

const MINI_W = CARD_W - 12;

/** Static measurement channels declared by the backend measurement artifact. */
export function IndicatorStack({ indicators }: { indicators: Indicator[] }) {
  if (indicators.length === 0) return null;
  return (
    <>
      <text x={6} y={CARD_H + 11} fontSize={8} fontWeight={600} fill={DAG_COLORS.muted}>
        {`measurement channels (${indicators.length})`}
      </text>
      {indicators.map((indicator, index) => (
        <g
          key={indicator.name}
          transform={`translate(6,${CARD_H + ISTACK_TOP + index * (MINI_H + IGAP)})`}
        >
          <rect
            width={MINI_W}
            height={MINI_H}
            rx={6}
            fill="#fff"
            stroke="#e6e9ee"
            strokeWidth={1}
          />
          <circle cx={8} cy={MINI_H / 2} r={2.2} fill={DAG_COLORS.realized} />
          <text x={15} y={14} fontSize={7.8} fontWeight={600} fill="#3a3f47">
            {indicator.name.replaceAll("_", " ")}
          </text>
          <text
            x={MINI_W - 6}
            y={14}
            textAnchor="end"
            fontSize={6.8}
            fontFamily="ui-monospace, monospace"
            fill={DAG_COLORS.muted}
          >
            {indicator.measurement_dtype}
          </text>
        </g>
      ))}
    </>
  );
}
