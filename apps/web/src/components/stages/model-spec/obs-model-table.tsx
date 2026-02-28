"use client";

import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { likelihoodLine, priorLatex } from "@/lib/utils/ssm-latex";
import type { LikelihoodSpec, ParameterSpec, PriorProposal } from "@causal-ssm/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import katex from "katex";

// ── helpers ──────────────────────────────────────────────

function inlineKatex(latex: string): string {
  return katex.renderToString(latex, { displayMode: false, throwOnError: false, strict: false });
}

// ── row type ─────────────────────────────────────────────

interface ObsModelRow {
  variable: string;
  construct: string | undefined;
  equationLatex: string;
  loadingFixed: boolean;
  priors: PriorProposal[];
}

// ── columns ──────────────────────────────────────────────

const col = createColumnHelper<ObsModelRow>();

const columns = [
  col.accessor("variable", {
    header: "Variable",
    cell: (info) => <span className="font-medium font-mono text-xs">{info.getValue()}</span>,
    meta: { mono: true },
  }),
  col.accessor("construct", {
    header: "Latent",
    cell: (info) => {
      const v = info.getValue();
      return v ? (
        <span className="font-mono text-xs">{v}</span>
      ) : (
        <span className="text-muted-foreground">—</span>
      );
    },
    meta: { mono: true },
  }),
  col.display({
    id: "equation",
    header: "Equation",
    cell: ({ row }) => (
      // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
      <span dangerouslySetInnerHTML={{ __html: inlineKatex(row.original.equationLatex) }} />
    ),
    enableSorting: false,
  }),
  col.accessor("loadingFixed", {
    header: () => (
      <HeaderWithTooltip
        label="Loading"
        tooltip="Whether the factor loading λ is fixed to 1 (reference indicator for scale identification) or freely estimated with a prior."
      />
    ),
    cell: ({ row }) => {
      const { construct, loadingFixed } = row.original;
      if (!construct) return <span className="text-muted-foreground">—</span>;
      return loadingFixed ? (
        // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
        <span dangerouslySetInnerHTML={{ __html: inlineKatex("= 1") }} />
      ) : (
        <span className="text-muted-foreground">estimated</span>
      );
    },
  }),
  col.display({
    id: "priors",
    header: "Priors",
    cell: ({ row }) => {
      const { priors } = row.original;
      if (priors.length === 0) return <span className="text-muted-foreground">—</span>;
      return (
        <div className="space-y-1">
          {priors.map((p) => (
            <div
              key={p.parameter}
              className="text-muted-foreground"
              // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
              dangerouslySetInnerHTML={{ __html: inlineKatex(priorLatex(p)) }}
            />
          ))}
        </div>
      );
    },
    enableSorting: false,
  }),
] as ColumnDef<ObsModelRow, unknown>[];

// ── component ────────────────────────────────────────────

export function ObsModelTable({
  likelihoods,
  parameters,
  priors,
  indicatorConstructMap,
}: {
  likelihoods: LikelihoodSpec[];
  parameters: ParameterSpec[];
  priors: PriorProposal[];
  indicatorConstructMap?: Record<string, string>;
}) {
  const rows: ObsModelRow[] = likelihoods.map((lik) => {
    const construct = indicatorConstructMap?.[lik.variable];
    const v = lik.variable;
    const hasLoadingParam = construct
      ? parameters.some((p) => p.role === "loading" && p.name === `lambda_${v}_${construct}`)
      : false;
    const loadingFixed = !!construct && !hasLoadingParam;
    const obsPriors = priors.filter(
      (p) =>
        p.parameter === `lambda_${v}_${construct}` ||
        p.parameter === `sigma_${v}` ||
        p.parameter === `sigma_obs_${v}` ||
        p.parameter === `phi_${v}`,
    );
    return {
      variable: v,
      construct,
      equationLatex: likelihoodLine(lik, construct).replace(/&/g, ""),
      loadingFixed,
      priors: obsPriors,
    };
  });

  return <InfoTable columns={columns} data={rows} estimateRowHeight={48} />;
}
