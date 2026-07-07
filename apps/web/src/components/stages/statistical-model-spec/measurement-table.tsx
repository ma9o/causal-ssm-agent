"use client";

import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatNumber } from "@/lib/utils/format";
import type { LikelihoodSpec, Stage4LikelihoodDiagnostics } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { scaleLinear } from "d3-scale";
import { curveMonotoneX, line } from "d3-shape";
import katex from "katex";
import { ExternalLink } from "lucide-react";
import { type MouseEvent, memo, useMemo, useState } from "react";
import { SparklineTooltip } from "./sparkline-tooltip";

// ── Row type ──────────────────────────────────────────────

interface MeasurementRow {
  likelihood: LikelihoodSpec;
  diagnostics?: Stage4LikelihoodDiagnostics;
  priorSamples?: number[];
}

interface DisplayBin {
  binCenter: number;
  count: number;
  binStart: number;
  binEnd: number;
}

const MAX_RENDER_BINS = 20;
const MEASUREMENT_CHART_WIDTH = 192;
const MEASUREMENT_CHART_HEIGHT = 80;
const MEASUREMENT_CHART_MARGIN = { top: 4, right: 6, bottom: 15, left: 4 };
const priorSeriesCache = new WeakMap<
  number[],
  Map<string, Array<{ binCenter: number; prior: number }>>
>();

function capHistogramBins(
  bins: Array<{ binCenter: number; count: number }>,
  maxBins = MAX_RENDER_BINS,
): DisplayBin[] {
  if (bins.length <= maxBins) {
    return bins.map((bin) => ({
      binCenter: bin.binCenter,
      count: bin.count,
      binStart: bin.binCenter,
      binEnd: bin.binCenter,
    }));
  }

  const groupSize = Math.ceil(bins.length / maxBins);
  const grouped: DisplayBin[] = [];

  for (let startIndex = 0; startIndex < bins.length; startIndex += groupSize) {
    const slice = bins.slice(startIndex, startIndex + groupSize);
    const binStart = slice[0].binCenter;
    const binEnd = slice[slice.length - 1].binCenter;
    grouped.push({
      binCenter: (binStart + binEnd) / 2,
      count: slice.reduce((sum, bin) => sum + bin.count, 0),
      binStart,
      binEnd,
    });
  }

  return grouped;
}

function binPriorSamples(
  priorSamples: number[],
  dataBins: DisplayBin[],
  nData: number,
  isDiscrete: boolean,
): Array<{ binCenter: number; prior: number }> {
  if (priorSamples.length === 0 || dataBins.length === 0) return [];

  if (isDiscrete) {
    const counts = new Array(dataBins.length).fill(0);
    for (const v of priorSamples) {
      const idx = dataBins.findIndex((bin) => v >= bin.binStart && v <= bin.binEnd);
      if (idx >= 0) {
        counts[idx]++;
      }
    }
    const scale = nData / priorSamples.length;
    return dataBins.map((bin, index) => ({
      binCenter: bin.binCenter,
      prior: counts[index] * scale,
    }));
  }

  const counts = new Array(dataBins.length).fill(0);
  const hasExplicitRanges = dataBins.some((bin) => bin.binStart !== bin.binEnd);

  if (!hasExplicitRanges) {
    if (dataBins.length < 2) return [];
    const binWidth = dataBins[1].binCenter - dataBins[0].binCenter;
    const firstEdge = dataBins[0].binCenter - binWidth / 2;

    for (const v of priorSamples) {
      const idx = Math.min(
        Math.max(Math.floor((v - firstEdge) / binWidth), 0),
        dataBins.length - 1,
      );
      counts[idx]++;
    }
  } else {
    for (const v of priorSamples) {
      const idx = dataBins.findIndex((bin, index) =>
        index === dataBins.length - 1
          ? v >= bin.binStart && v <= bin.binEnd
          : v >= bin.binStart && v < bin.binEnd,
      );
      if (idx >= 0) {
        counts[idx]++;
      }
    }
  }

  const scale = nData / priorSamples.length;
  return dataBins.map((bin, index) => ({
    binCenter: bin.binCenter,
    prior: counts[index] * scale,
  }));
}

function priorCacheKey(dataBins: DisplayBin[], nData: number, isDiscrete: boolean): string {
  return [
    isDiscrete ? "discrete" : "continuous",
    String(nData),
    dataBins.map((bin) => `${bin.binStart}:${bin.binEnd}:${bin.binCenter}`).join("|"),
  ].join("::");
}

function getCachedPriorSamples(
  priorSamples: number[],
  dataBins: DisplayBin[],
  nData: number,
  isDiscrete: boolean,
): Array<{ binCenter: number; prior: number }> {
  if (priorSamples.length === 0 || dataBins.length === 0) {
    return [];
  }

  const key = priorCacheKey(dataBins, nData, isDiscrete);
  const cachedForSamples = priorSeriesCache.get(priorSamples);
  const cachedSeries = cachedForSamples?.get(key);
  if (cachedSeries) {
    return cachedSeries;
  }

  const computed = binPriorSamples(priorSamples, dataBins, nData, isDiscrete);
  if (cachedForSamples) {
    cachedForSamples.set(key, computed);
  } else {
    priorSeriesCache.set(priorSamples, new Map([[key, computed]]));
  }
  return computed;
}

interface MeasurementChartPoint extends DisplayBin {
  prior?: number;
}

function measurementXDomain(data: MeasurementChartPoint[]): [number, number] {
  const min = Math.min(...data.map((bin) => Math.min(bin.binStart, bin.binCenter)));
  const max = Math.max(...data.map((bin) => Math.max(bin.binEnd, bin.binCenter)));

  if (min === max) {
    return [min - 0.5, max + 0.5];
  }

  return [min, max];
}

// ── Link label helper ─────────────────────────────────────

function linkLabel(link: string): string {
  switch (link) {
    case "identity":
      return "E[y] = \u03BC";
    case "log":
      return "E[y] = exp(\u03BC)";
    case "logit":
      return "E[y] = \u03C3(\u03BC)";
    case "probit":
      return "E[y] = \u03A6(\u03BC)";
    default:
      return "g\u207B\u00B9(\u03BC)";
  }
}

// ── Inline chart ──────────────────────────────────────────

const MeasurementSparkline = memo(
  function MeasurementSparkline({ row }: { row: MeasurementRow }) {
    const nObs = row.diagnostics?.profile?.n_obs ?? 0;
    const isDiscrete =
      row.likelihood.distribution === "poisson" ||
      row.likelihood.distribution === "bernoulli" ||
      row.likelihood.distribution === "negative_binomial" ||
      row.likelihood.distribution === "ordered_logistic";
    const bins = useMemo(
      () => capHistogramBins(row.diagnostics?.histogram ?? []),
      [row.diagnostics?.histogram],
    );

    const hasHistogram = bins.length > 0 && nObs > 0;

    const prior = useMemo(
      () =>
        hasHistogram && row.priorSamples && row.priorSamples.length > 0
          ? getCachedPriorSamples(row.priorSamples, bins, nObs, isDiscrete)
          : [],
      [bins, hasHistogram, isDiscrete, nObs, row.priorSamples],
    );

    const hasPrior = prior.length > 0;

    const chartData: MeasurementChartPoint[] = useMemo(() => {
      if (!hasHistogram) {
        return [];
      }
      const priorByCenter = new Map(prior.map((entry) => [entry.binCenter, entry.prior]));
      return bins.map((bin) => ({
        ...bin,
        ...(hasPrior ? { prior: priorByCenter.get(bin.binCenter) ?? 0 } : {}),
      }));
    }, [bins, hasHistogram, hasPrior, prior]);

    const [hoverIndex, setHoverIndex] = useState<number | null>(null);

    if (!hasHistogram) {
      return <span className="text-xs text-muted-foreground">--</span>;
    }

    const plotLeft = MEASUREMENT_CHART_MARGIN.left;
    const plotRight = MEASUREMENT_CHART_WIDTH - MEASUREMENT_CHART_MARGIN.right;
    const plotTop = MEASUREMENT_CHART_MARGIN.top;
    const plotBottom = MEASUREMENT_CHART_HEIGHT - MEASUREMENT_CHART_MARGIN.bottom;
    const [xMin, xMax] = measurementXDomain(chartData);
    const xScale = scaleLinear().domain([xMin, xMax]).range([plotLeft, plotRight]);
    const maxY = Math.max(
      1,
      ...chartData.map((bin) => bin.count),
      ...chartData.map((bin) => bin.prior ?? 0),
    );
    const yScale = scaleLinear().domain([0, maxY]).nice().range([plotBottom, plotTop]);
    const gridTicks = yScale.ticks(3).filter((tick) => tick > 0);
    const defaultBarWidth = Math.max(2, (plotRight - plotLeft) / chartData.length - 1);
    const priorPath = hasPrior
      ? line<MeasurementChartPoint>()
          .x((point) => xScale(point.binCenter))
          .y((point) => yScale(point.prior ?? 0))
          .curve(curveMonotoneX)(chartData)
      : null;
    const xLabels = xMin === xMax ? [xMin] : [xMin, xMax];

    const hovered =
      hoverIndex != null && hoverIndex < chartData.length ? chartData[hoverIndex] : null;

    const handleMove = (event: MouseEvent<HTMLDivElement>) => {
      const rect = event.currentTarget.getBoundingClientRect();
      if (rect.width === 0) return;
      const pointerX = ((event.clientX - rect.left) / rect.width) * MEASUREMENT_CHART_WIDTH;
      let nearest = 0;
      let nearestDist = Number.POSITIVE_INFINITY;
      for (let index = 0; index < chartData.length; index++) {
        const dist = Math.abs(xScale(chartData[index].binCenter) - pointerX);
        if (dist < nearestDist) {
          nearestDist = dist;
          nearest = index;
        }
      }
      setHoverIndex(nearest);
    };

    return (
      <div
        className="h-20 w-48 cursor-crosshair"
        onMouseMove={handleMove}
        onMouseLeave={() => setHoverIndex(null)}
      >
        <svg
          className="h-full w-full"
          viewBox={`0 0 ${MEASUREMENT_CHART_WIDTH} ${MEASUREMENT_CHART_HEIGHT}`}
          role="img"
          aria-label="Empirical data histogram overlaid with prior predictive line"
        >
          {gridTicks.map((tick) => (
            <line
              key={tick}
              x1={plotLeft}
              x2={plotRight}
              y1={yScale(tick)}
              y2={yScale(tick)}
              stroke="var(--muted)"
              strokeDasharray="3 3"
            />
          ))}
          {chartData.map((bin, index) => {
            const hasRange = bin.binStart !== bin.binEnd;
            const rangeWidth = hasRange
              ? xScale(bin.binEnd) - xScale(bin.binStart)
              : defaultBarWidth;
            const barWidth = Math.max(
              2,
              Math.min(isDiscrete ? 14 : Number.POSITIVE_INFINITY, rangeWidth - 1),
            );
            const x = hasRange ? xScale(bin.binStart) : xScale(bin.binCenter) - barWidth / 2;
            const y = yScale(bin.count);
            return (
              <rect
                key={`${bin.binStart}:${bin.binEnd}:${bin.binCenter}`}
                x={x}
                y={y}
                width={barWidth}
                height={plotBottom - y}
                fill="var(--muted-foreground)"
                opacity={hoverIndex === index ? 0.55 : 0.3}
              />
            );
          })}
          {priorPath && (
            <path d={priorPath} fill="none" stroke="var(--primary)" strokeWidth={1.5} />
          )}
          <line
            x1={plotLeft}
            x2={plotRight}
            y1={plotBottom}
            y2={plotBottom}
            stroke="var(--border)"
          />
          {xLabels.map((value, index) => (
            <text
              key={value}
              x={index === 0 ? plotLeft : plotRight}
              y={MEASUREMENT_CHART_HEIGHT - 2}
              textAnchor={index === 0 ? "start" : "end"}
              fill="var(--muted-foreground)"
              fontSize={9}
            >
              {formatNumber(value, 1)}
            </text>
          ))}
          {hovered && (
            <g pointerEvents="none">
              <line
                x1={xScale(hovered.binCenter)}
                x2={xScale(hovered.binCenter)}
                y1={plotTop}
                y2={plotBottom}
                stroke="var(--muted-foreground)"
                strokeWidth={1}
                opacity={0.5}
              />
              <circle
                cx={xScale(hovered.binCenter)}
                cy={yScale(hovered.count)}
                r={2.5}
                fill="var(--muted-foreground)"
              />
              {hasPrior && (
                <circle
                  cx={xScale(hovered.binCenter)}
                  cy={yScale(hovered.prior ?? 0)}
                  r={2.5}
                  fill="var(--primary)"
                />
              )}
              <SparklineTooltip
                anchorX={xScale(hovered.binCenter)}
                anchorY={yScale(hovered.count)}
                width={MEASUREMENT_CHART_WIDTH}
                height={MEASUREMENT_CHART_HEIGHT}
                lines={[
                  `x = ${formatNumber(hovered.binCenter, 2)}`,
                  `count = ${formatNumber(hovered.count, 1)}`,
                  ...(hasPrior ? [`prior ≈ ${formatNumber(hovered.prior ?? 0, 1)}`] : []),
                ]}
              />
            </g>
          )}
        </svg>
      </div>
    );
  },
  (previous, next) =>
    previous.row.diagnostics === next.row.diagnostics &&
    previous.row.priorSamples === next.row.priorSamples &&
    previous.row.likelihood.distribution === next.row.likelihood.distribution,
);

// ── Table columns ─────────────────────────────────────────

const col = createColumnHelper<MeasurementRow>();

const baseColumns: ColumnDef<MeasurementRow, unknown>[] = [
  col.display({
    id: "variable",
    header: "Variable",
    cell: ({ row }) => (
      <span className="font-medium font-mono text-xs">{row.original.likelihood.variable}</span>
    ),
  }),
  col.display({
    id: "distribution",
    header: "Distribution",
    cell: ({ row }) => <Badge variant="outline">{row.original.likelihood.distribution}</Badge>,
  }),
  col.display({
    id: "link",
    header: "Link",
    cell: ({ row }) => <Badge variant="secondary">{linkLabel(row.original.likelihood.link)}</Badge>,
  }),
  col.display({
    id: "chart",
    header: () => (
      <span className="inline-flex items-center gap-1">
        Data vs Prior
        <StatTooltip explanation="Empirical data histogram (grey bars) overlaid with marginal prior predictive samples (line). Compare to check whether priors imply a plausible data scale." />
      </span>
    ),
    cell: ({ row }) => <MeasurementSparkline row={row.original} />,
  }),
  col.display({
    id: "stats",
    header: "Stats",
    cell: ({ row }) => {
      const profile = row.original.diagnostics?.profile;
      if (!profile || profile.n_obs === 0 || profile.mean == null) {
        return <span className="text-xs text-muted-foreground">--</span>;
      }
      const latex = `n=${profile.n_obs} \\\\[2pt] \\hat{\\mu}=${formatNumber(profile.mean, 2)}`;
      return (
        <span
          className="text-xs text-muted-foreground"
          // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
          dangerouslySetInnerHTML={{
            __html: katex.renderToString(latex, {
              displayMode: false,
              throwOnError: false,
              strict: false,
            }),
          }}
        />
      );
    },
  }),
  col.display({
    id: "reasoning",
    header: "Reasoning",
    cell: ({ row }) => (
      <span className="max-w-xs whitespace-normal text-xs text-muted-foreground">
        {row.original.likelihood.reasoning}
      </span>
    ),
  }),
  col.display({
    id: "sources",
    header: () => (
      <HeaderWithTooltip
        label="Sources"
        tooltip="Literature sources supporting this likelihood distribution choice. Click to open."
      />
    ),
    cell: ({ row }) => {
      const sources = row.original.likelihood.sources;
      if (!sources || sources.length === 0) {
        return <span className="text-xs text-muted-foreground">--</span>;
      }
      return (
        <div className="flex items-center gap-1.5">
          {sources.map((source, i) => (
            <Tooltip
              key={`source-${
                // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
                i
              }`}
            >
              <TooltipTrigger>
                {source.url ? (
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-0.5 text-primary hover:underline"
                  >
                    <Badge variant="secondary" className="cursor-pointer text-[10px] px-1.5">
                      {i + 1}
                      <ExternalLink className="ml-0.5 h-2.5 w-2.5" />
                    </Badge>
                  </a>
                ) : (
                  <Badge variant="secondary" className="text-[10px] px-1.5">
                    {i + 1}
                  </Badge>
                )}
              </TooltipTrigger>
              <TooltipContent>
                <div className="max-w-xs text-xs">
                  <p className="font-medium">{source.title}</p>
                  <p className="text-muted-foreground">{source.snippet}</p>
                </div>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      );
    },
    meta: { align: "center" },
  }),
];

// ── Exported component ────────────────────────────────────

export function MeasurementTable({
  likelihoods,
  diagnostics,
  priorPredictiveSamples,
}: {
  likelihoods: LikelihoodSpec[];
  diagnostics: Record<string, Stage4LikelihoodDiagnostics | undefined>;
  priorPredictiveSamples?: Record<string, number[]>;
}) {
  const rows: MeasurementRow[] = useMemo(
    () =>
      likelihoods.map((lik) => ({
        likelihood: lik,
        diagnostics: diagnostics[lik.variable],
        priorSamples: priorPredictiveSamples?.[lik.variable],
      })),
    [likelihoods, diagnostics, priorPredictiveSamples],
  );

  const columns = baseColumns;

  return <InfoTable columns={columns} data={rows} estimateRowHeight={88} />;
}
