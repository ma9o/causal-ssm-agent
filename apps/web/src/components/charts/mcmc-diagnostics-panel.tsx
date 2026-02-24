"use client";

import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { createColumnHelper, type ColumnDef } from "@tanstack/react-table";
import { formatNumber } from "@/lib/utils/format";
import type { MCMCDiagnostics, MCMCParamDiagnostic } from "@causal-ssm/api-types";
import { AlertTriangle } from "lucide-react";

interface MCMCDiagnosticsPanelProps {
  diagnostics: MCMCDiagnostics;
}

function rhatSeverity(value: number | number[]): "fail" | "warn" | undefined {
  const v = Array.isArray(value) ? Math.max(...value) : value;
  if (v >= 1.1) return "fail";
  if (v >= 1.01) return "warn";
  return undefined;
}

function essSeverity(
  value: number | number[] | undefined,
  nSamples: number | null,
): "fail" | "warn" | undefined {
  if (value == null) return undefined;
  const v = Array.isArray(value) ? Math.min(...value) : value;
  const total = nSamples ?? 1000;
  const ratio = v / total;
  if (ratio <= 0.1) return "fail";
  if (ratio <= 0.5) return "warn";
  return undefined;
}

const col = createColumnHelper<MCMCParamDiagnostic>();

export function MCMCDiagnosticsPanel({ diagnostics }: MCMCDiagnosticsPanelProps) {
  const hasDivergences = diagnostics.num_divergences > 0;
  const hasEssTail = diagnostics.per_parameter.some((p) => p.ess_tail != null);
  const hasMcse = diagnostics.per_parameter.some((p) => p.mcse_mean != null);

  const columns = useMemo<ColumnDef<MCMCParamDiagnostic, unknown>[]>(() => {
    const cols = [
      col.accessor("parameter", {
        header: "Parameter",
        cell: (info) => (
          <span className="font-medium">{info.getValue()}</span>
        ),
        meta: { mono: true },
      }),
      col.accessor("r_hat", {
        header: () => (
          <HeaderWithTooltip
            label="R-hat"
            tooltip="Potential scale reduction factor. Values near 1.0 indicate convergence. Worry above 1.01."
          />
        ),
        cell: (info) => {
          const v = info.getValue();
          const val = Array.isArray(v) ? Math.max(...v) : v;
          return formatNumber(val, 3);
        },
        meta: {
          align: "right",
          mono: true,
          severity: (v: number | number[]) => rhatSeverity(v),
        },
      }),
      col.accessor("ess_bulk", {
        header: () => (
          <HeaderWithTooltip
            label="ESS (bulk)"
            tooltip="Effective sample size for bulk of the distribution. Higher is better. Worry if < 100 per chain."
          />
        ),
        cell: (info) => {
          const v = info.getValue();
          if (v == null) return <span className="text-muted-foreground">—</span>;
          const val = Array.isArray(v) ? Math.min(...v) : v;
          return formatNumber(val, 0);
        },
        meta: {
          align: "right",
          mono: true,
          severity: (v: number | number[] | undefined) =>
            essSeverity(v, diagnostics.num_samples ?? null),
        },
      }),
    ] as ColumnDef<MCMCParamDiagnostic, unknown>[];

    if (hasEssTail) {
      cols.push(
        col.accessor("ess_tail", {
          header: () => (
            <HeaderWithTooltip
              label="ESS (tail)"
              tooltip="Effective sample size for the tails (5th/95th percentiles). Important for credible interval reliability."
            />
          ),
          cell: (info) => {
            const v = info.getValue();
            if (v == null) return <span className="text-muted-foreground">—</span>;
            const val = Array.isArray(v) ? Math.min(...v) : v;
            return formatNumber(val, 0);
          },
          meta: {
            align: "right",
            mono: true,
            severity: (v: number | number[] | null | undefined) =>
              essSeverity(v ?? undefined, diagnostics.num_samples ?? null),
          },
        }) as ColumnDef<MCMCParamDiagnostic, unknown>,
      );
    }

    if (hasMcse) {
      cols.push(
        col.accessor("mcse_mean", {
          header: () => (
            <HeaderWithTooltip
              label="MCSE"
              tooltip="Monte Carlo standard error of the mean. Should be small relative to the posterior standard deviation."
            />
          ),
          cell: (info) => {
            const v = info.getValue();
            if (v == null) return <span className="text-muted-foreground">—</span>;
            const val = Array.isArray(v) ? Math.max(...v) : v;
            return formatNumber(val, 4);
          },
          meta: { align: "right", mono: true },
        }) as ColumnDef<MCMCParamDiagnostic, unknown>,
      );
    }

    return cols;
  }, [hasEssTail, hasMcse, diagnostics.num_samples]);

  return (
    <div className="space-y-3">
      {/* Sampler-level summary */}
      <div className="flex flex-wrap gap-2">
        {diagnostics.num_chains != null && (
          <Badge variant="secondary">{diagnostics.num_chains} chains</Badge>
        )}
        {diagnostics.num_samples != null && (
          <Badge variant="secondary">{diagnostics.num_samples.toLocaleString()} samples</Badge>
        )}
        <Badge variant={hasDivergences ? "destructive" : "success"}>
          {hasDivergences && <AlertTriangle className="mr-1 h-3 w-3" />}
          {diagnostics.num_divergences} divergence{diagnostics.num_divergences !== 1 && "s"}
          {hasDivergences && ` (${formatNumber(diagnostics.divergence_rate * 100, 1)}%)`}
        </Badge>
        <Badge variant="secondary">
          tree depth: {formatNumber(diagnostics.tree_depth_mean, 1)} avg, {diagnostics.tree_depth_max} max
        </Badge>
        <Badge variant="secondary">
          accept: {formatNumber(diagnostics.accept_prob_mean * 100, 1)}%
        </Badge>
      </div>

      {/* Per-parameter table */}
      {diagnostics.per_parameter.length > 0 && (
        <InfoTable columns={columns} data={diagnostics.per_parameter} />
      )}
    </div>
  );
}
