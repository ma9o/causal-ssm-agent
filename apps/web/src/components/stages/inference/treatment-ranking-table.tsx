"use client";

import type { TreatmentEffect } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { ChevronDown, ChevronUp } from "lucide-react";
import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { formatNumber } from "@/lib/utils/format";
import {
  drawsCI,
  meanDraws,
  peakEffect,
  probPositive,
  timeToPeak,
} from "@/lib/utils/treatment-effect-stats";
import { ManifestProjection, PosteriorHistogram } from "./treatment-effect-visuals";

const col = createColumnHelper<TreatmentEffect>();

export function TreatmentRankingTable({ results }: { results: TreatmentEffect[] }) {
  const [selectedTreatment, setSelectedTreatment] = useState<string | null>();

  const sorted = useMemo(
    () =>
      [...results].sort(
        (a, b) =>
          Math.abs(meanDraws(b.posterior_draws) ?? 0) - Math.abs(meanDraws(a.posterior_draws) ?? 0),
      ),
    [results],
  );

  const columns = useMemo(
    () => [
      col.accessor("treatment", {
        header: "Treatment",
        cell: (info) => <span className="font-medium">{info.getValue()}</span>,
      }),
      col.display({
        id: "effect_size",
        header: () => (
          <HeaderWithTooltip
            label="τ̂"
            tooltip="Estimated total interventional treatment effect via do-operator steady-state intervention. Positive values indicate the treatment increases the outcome."
            className="font-mono"
          />
        ),
        cell: ({ row }) => {
          const v = meanDraws(row.original.posterior_draws);
          return v === null ? "—" : formatNumber(v);
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "ci_95",
        header: () => (
          <HeaderWithTooltip
            label="95% CI"
            tooltip="95% credible interval computed from the posterior draws (2.5th–97.5th percentile)."
          />
        ),
        cell: ({ row }) => {
          const ci = drawsCI(row.original.posterior_draws);
          return ci === null ? "—" : `[${formatNumber(ci.lower)}, ${formatNumber(ci.upper)}]`;
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "prob_positive",
        header: () => (
          <HeaderWithTooltip
            label="P(τ > 0)"
            tooltip="Posterior probability that the total intervention effect is positive."
            className="font-mono"
          />
        ),
        cell: ({ row }) => {
          const probability = probPositive(row.original.posterior_draws);
          return probability === null ? "—" : `${Math.round(probability * 100)}%`;
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "peak",
        header: () => (
          <HeaderWithTooltip
            label="Peak"
            tooltip="Maximum absolute forward-simulated effect over the default Stage 6 horizon."
          />
        ),
        cell: ({ row }) => {
          const peak = peakEffect(row.original);
          return peak === null ? "—" : formatNumber(peak);
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "time_to_peak",
        header: () => (
          <HeaderWithTooltip
            label="t → peak"
            tooltip="Days from intervention onset to the peak absolute effect."
            className="font-mono"
          />
        ),
        cell: ({ row }) => {
          const days = timeToPeak(row.original);
          return days === null ? "—" : `${formatNumber(days, 1)}d`;
        },
        meta: {
          align: "right",
          mono: true,
        },
      }),
      col.display({
        id: "posterior",
        header: () => (
          <HeaderWithTooltip
            label="Posterior"
            tooltip="Full posterior distribution of the treatment effect. The vertical dashed line marks zero — draws to the right indicate a positive effect."
          />
        ),
        cell: ({ row }) => {
          const draws = row.original.posterior_draws;
          return draws && draws.length > 0 ? (
            <PosteriorHistogram draws={draws} mean={meanDraws(draws)} />
          ) : (
            "—"
          );
        },
        meta: {
          align: "right",
        },
      }),
      col.display({
        id: "projection",
        header: () => (
          <HeaderWithTooltip
            label="Indicators"
            tooltip="Expandable indicator-level projection of the outcome effect through the measurement loadings."
          />
        ),
        cell: ({ row }) => {
          const manifestCount = Object.keys(row.original.manifest_effects ?? {}).length;
          if (manifestCount === 0) {
            return <span className="text-muted-foreground">—</span>;
          }
          const isSelected = row.original.treatment === selectedTreatment;
          return (
            <Button
              variant="ghost"
              size="xs"
              onClick={() => setSelectedTreatment(isSelected ? null : row.original.treatment)}
            >
              {isSelected ? <ChevronUp className="size-3" /> : <ChevronDown className="size-3" />}
              {isSelected ? "Hide" : "Show"} {manifestCount}
            </Button>
          );
        },
        meta: {
          align: "right",
        },
      }),
    ],
    [selectedTreatment],
  );

  return (
    <InfoTable
      columns={columns as ColumnDef<TreatmentEffect, unknown>[]}
      data={sorted}
      compact
      estimateRowHeight={68}
      rowClassName={(row, index) => {
        if (row.treatment === selectedTreatment) {
          return "bg-muted/40";
        }
        return index === 0 ? "bg-emerald-500/10" : undefined;
      }}
      isRowExpanded={(row) =>
        row.treatment === selectedTreatment && Object.keys(row.manifest_effects ?? {}).length > 0
      }
      renderExpandedRow={(row) => (
        <ManifestProjection manifestEffects={row.manifest_effects} className="px-6 py-2.5" />
      )}
    />
  );
}
