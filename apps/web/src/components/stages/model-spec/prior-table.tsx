"use client";

import { Badge } from "@/components/ui/badge";
import { HeaderWithTooltip, InfoTable } from "@/components/ui/info-table";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { evaluatePdf } from "@/lib/utils/distributions";
import { formatNumber } from "@/lib/utils/format";
import type { ParameterSpec, PriorProposal } from "@nof1-causal-lab/api-types";
import { type ColumnDef, createColumnHelper } from "@tanstack/react-table";
import { scaleLinear } from "d3-scale";
import { area, curveMonotoneX, line } from "d3-shape";
import { ExternalLink } from "lucide-react";
import { type MouseEvent, useMemo, useState } from "react";
import { SparklineTooltip } from "./sparkline-tooltip";

type PriorRow = PriorProposal;

const col = createColumnHelper<PriorRow>();
const DENSITY_CHART_WIDTH = 144;
const DENSITY_CHART_HEIGHT = 64;
const DENSITY_CHART_MARGIN = { top: 4, right: 5, bottom: 14, left: 3 };

interface DensityPoint {
  x: number;
  y: number;
}

function densityPoints(points: PriorRow["density_points"]): DensityPoint[] {
  return (points ?? []).flatMap((point) =>
    typeof point.x === "number" && typeof point.y === "number"
      ? [{ x: point.x, y: point.y }]
      : [],
  );
}

/** Compact inline density chart with axes. */
function DensitySparkline({ prior }: { prior: PriorRow }) {
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const data = useMemo(
    () =>
      densityPoints(
        prior.density_points ??
          evaluatePdf(prior.distribution, prior.params as Record<string, number>, 60),
      ),
    [prior],
  );

  if (data.length === 0) {
    return <span className="text-xs text-muted-foreground">--</span>;
  }

  const plotLeft = DENSITY_CHART_MARGIN.left;
  const plotRight = DENSITY_CHART_WIDTH - DENSITY_CHART_MARGIN.right;
  const plotTop = DENSITY_CHART_MARGIN.top;
  const plotBottom = DENSITY_CHART_HEIGHT - DENSITY_CHART_MARGIN.bottom;
  const xMin = Math.min(...data.map((point) => point.x));
  const xMax = Math.max(...data.map((point) => point.x));
  const xPadding = xMin === xMax ? 0.5 : 0;
  const xScale = scaleLinear()
    .domain([xMin - xPadding, xMax + xPadding])
    .range([plotLeft, plotRight]);
  const maxY = Math.max(1e-9, ...data.map((point) => point.y));
  const yScale = scaleLinear().domain([0, maxY]).nice().range([plotBottom, plotTop]);
  const areaPath = area<DensityPoint>()
    .x((point) => xScale(point.x))
    .y0(plotBottom)
    .y1((point) => yScale(point.y))
    .curve(curveMonotoneX)(data);
  const linePath = line<DensityPoint>()
    .x((point) => xScale(point.x))
    .y((point) => yScale(point.y))
    .curve(curveMonotoneX)(data);
  const xLabels = xMin === xMax ? [xMin] : [xMin, xMax];

  const hovered = hoverIndex != null && hoverIndex < data.length ? data[hoverIndex] : null;

  const handleMove = (event: MouseEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    const pointerX = ((event.clientX - rect.left) / rect.width) * DENSITY_CHART_WIDTH;
    let nearest = 0;
    let nearestDist = Number.POSITIVE_INFINITY;
    for (let index = 0; index < data.length; index++) {
      const dist = Math.abs(xScale(data[index].x) - pointerX);
      if (dist < nearestDist) {
        nearestDist = dist;
        nearest = index;
      }
    }
    setHoverIndex(nearest);
  };

  return (
    <div
      className="h-16 w-36 cursor-crosshair"
      onMouseMove={handleMove}
      onMouseLeave={() => setHoverIndex(null)}
    >
      <svg
        className="h-full w-full"
        viewBox={`0 0 ${DENSITY_CHART_WIDTH} ${DENSITY_CHART_HEIGHT}`}
        role="img"
        aria-label={`Prior density for ${prior.parameter}`}
      >
        {areaPath && <path d={areaPath} fill="var(--primary)" opacity={0.15} />}
        {linePath && <path d={linePath} fill="none" stroke="var(--primary)" strokeWidth={1.5} />}
        <line x1={plotLeft} x2={plotRight} y1={plotBottom} y2={plotBottom} stroke="var(--border)" />
        {xLabels.map((value, index) => (
          <text
            key={value}
            x={index === 0 ? plotLeft : plotRight}
            y={DENSITY_CHART_HEIGHT - 2}
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
              x1={xScale(hovered.x)}
              x2={xScale(hovered.x)}
              y1={plotTop}
              y2={plotBottom}
              stroke="var(--muted-foreground)"
              strokeWidth={1}
              opacity={0.5}
            />
            <circle cx={xScale(hovered.x)} cy={yScale(hovered.y)} r={2.5} fill="var(--primary)" />
            <SparklineTooltip
              anchorX={xScale(hovered.x)}
              anchorY={yScale(hovered.y)}
              width={DENSITY_CHART_WIDTH}
              height={DENSITY_CHART_HEIGHT}
              lines={[`x = ${formatNumber(hovered.x, 2)}`, `p = ${formatNumber(hovered.y, 3)}`]}
            />
          </g>
        )}
      </svg>
    </div>
  );
}

const baseColumns = [
  col.accessor("parameter", {
    header: "Parameter",
    cell: (info) => <span className="font-medium font-mono text-xs">{info.getValue()}</span>,
  }),
  col.accessor("distribution", {
    header: "Distribution",
    cell: (info) => <Badge variant="outline">{info.getValue()}</Badge>,
  }),
  col.display({
    id: "params",
    header: "Params",
    cell: ({ row }) => {
      const params = row.original.params as Record<string, number>;
      return (
        <div className="flex flex-col gap-0.5 font-mono text-xs text-muted-foreground">
          {Object.entries(params).map(([k, v]) => (
            <span key={k}>
              {k}={formatNumber(v, 2)}
            </span>
          ))}
        </div>
      );
    },
  }),
  col.display({
    id: "density",
    header: "Density",
    cell: ({ row }) => <DensitySparkline prior={row.original} />,
  }),
  col.accessor("reasoning", {
    header: "Reasoning",
    cell: (info) => (
      <span className="max-w-xs whitespace-normal text-xs text-muted-foreground">{info.getValue()}</span>
    ),
  }),
  col.display({
    id: "sources",
    header: () => (
      <HeaderWithTooltip
        label="Sources"
        tooltip="Literature sources supporting this prior choice. Click to open."
      />
    ),
    cell: ({ row }) => {
      const sources = row.original.sources;
      if (sources.length === 0) {
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
                  {source.effect_size && (
                    <span className="text-muted-foreground">Effect: {source.effect_size}</span>
                  )}
                </div>
              </TooltipContent>
            </Tooltip>
          ))}
        </div>
      );
    },
    meta: { align: "center" },
  }),
] as ColumnDef<PriorRow, unknown>[];

export function PriorTable({
  priors,
}: { priors: PriorProposal[]; parameters?: ParameterSpec[] }) {
  return <InfoTable columns={baseColumns} data={priors} estimateRowHeight={72} />;
}
