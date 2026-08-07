"use client";

import { BadgeCheck, ChartSpline, Database, Link2, type LucideIcon, Rows3 } from "lucide-react";
import { useMemo } from "react";
import { IndicatorTable } from "@/components/analysis-widgets/measurement-structure/indicator-table";
import { PPCWarningsTable } from "@/components/analysis-widgets/posterior/ppc-warnings-table";
import MeasurementsView from "@/components/pipeline/output-views/measurements-view";
import RawDataView from "@/components/pipeline/output-views/raw-data-view";
import ValidationReportView from "@/components/pipeline/output-views/validation-report-view";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  MODEL_NODE_IDS,
  type ModelNodeId,
  PROTOTYPE_INDICATORS,
  PROTOTYPE_MEASUREMENTS,
  PROTOTYPE_NODE_META,
  PROTOTYPE_PPC_OVERLAYS,
  PROTOTYPE_PPC_TEST_STATS,
  PROTOTYPE_PPC_WARNINGS,
  PROTOTYPE_RAW_DATA,
  PROTOTYPE_VALIDATION_REPORT,
  type WorkspaceLayerId,
} from "./artifact-workspace-fixture";

interface DataCanvasProps {
  visibleLayers: ReadonlySet<WorkspaceLayerId>;
  selectedNode: ModelNodeId;
  onSelectNode: (node: ModelNodeId) => void;
}

function ExistingSurface({
  title,
  artifact,
  component,
  icon: Icon,
  className,
  children,
}: {
  title: string;
  artifact: string;
  component: string;
  icon: LucideIcon;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <section
      className={cn("min-w-0 overflow-hidden rounded-xl border bg-white shadow-sm", className)}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-slate-50/70 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Icon className="size-3.5 text-slate-500" />
          <span className="text-xs font-semibold text-slate-800">{title}</span>
          <span className="font-mono text-[8px] text-muted-foreground">{artifact}</span>
        </div>
        <Badge variant="outline" className="font-mono text-[8px]">
          {component}
        </Badge>
      </div>
      <div className="overflow-auto p-4">{children}</div>
    </section>
  );
}

function NoDirectIndicator({ selectedNode }: { selectedNode: ModelNodeId }) {
  return (
    <div className="rounded-lg border border-dashed bg-slate-50/60 p-5 text-center text-xs text-muted-foreground">
      {PROTOTYPE_NODE_META[selectedNode].label} has no direct indicator. It remains explicit in the
      causal model and is integrated out during fitting.
    </div>
  );
}

export function DataCanvas({ visibleLayers, selectedNode, onSelectNode }: DataCanvasProps) {
  const sourceVisible = visibleLayers.has("data.source");
  const mappingVisible = visibleLayers.has("data.mapping");
  const observationsVisible = visibleLayers.has("data.observations");
  const qualityVisible = visibleLayers.has("data.quality");
  const fitVisible = visibleLayers.has("data.fit");

  const selectedIndicators = useMemo(
    () => PROTOTYPE_INDICATORS.filter((indicator) => indicator.construct_name === selectedNode),
    [selectedNode],
  );
  const selectedIndicatorNames = useMemo(
    () => new Set(selectedIndicators.map((indicator) => indicator.name)),
    [selectedIndicators],
  );
  const selectedMeasurements = useMemo(
    () => ({
      ...PROTOTYPE_MEASUREMENTS,
      per_indicator_counts: Object.fromEntries(
        Object.entries(PROTOTYPE_MEASUREMENTS.per_indicator_counts).filter(([indicator]) =>
          selectedIndicatorNames.has(indicator),
        ),
      ),
      combined_extractions_sample: PROTOTYPE_MEASUREMENTS.combined_extractions_sample.filter(
        (observation) => selectedIndicatorNames.has(observation.indicator),
      ),
    }),
    [selectedIndicatorNames],
  );
  const selectedValidation = useMemo(
    () => ({
      ...PROTOTYPE_VALIDATION_REPORT,
      indicators: Object.fromEntries(
        Object.entries(PROTOTYPE_VALIDATION_REPORT.indicators).filter(([indicator]) =>
          selectedIndicatorNames.has(indicator),
        ),
      ),
    }),
    [selectedIndicatorNames],
  );
  const selectedPpcWarnings = PROTOTYPE_PPC_WARNINGS.filter((warning) =>
    selectedIndicatorNames.has(warning.variable),
  );
  const selectedPpcStats = PROTOTYPE_PPC_TEST_STATS.filter((stat) =>
    selectedIndicatorNames.has(stat.variable),
  );
  const selectedPpcOverlays = PROTOTYPE_PPC_OVERLAYS.filter((overlay) =>
    selectedIndicatorNames.has(overlay.variable),
  );
  const observableNodes = MODEL_NODE_IDS.filter((node) =>
    PROTOTYPE_INDICATORS.some((indicator) => indicator.construct_name === node),
  );

  return (
    <div className="min-h-0 bg-slate-50/70">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b bg-white px-4 py-2.5">
        <div className="flex items-center gap-2">
          <Link2 className="size-3.5 text-blue-600" />
          <span className="text-xs font-medium text-slate-700">Construct-linked evidence</span>
        </div>
        <div className="flex flex-wrap items-center gap-1">
          {observableNodes.map((node) => (
            <button
              key={node}
              type="button"
              onClick={() => onSelectNode(node)}
              aria-pressed={selectedNode === node}
              className={cn(
                "rounded-md px-2 py-1 text-[9px] font-medium transition-colors",
                selectedNode === node
                  ? "bg-slate-900 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200",
              )}
            >
              {PROTOTYPE_NODE_META[node].label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid min-w-0 gap-3 p-3 xl:grid-cols-2">
        {sourceVisible ? (
          <ExistingSurface
            title="Source records"
            artifact="raw_data"
            component="RawDataView"
            icon={Database}
            className="xl:col-span-2"
          >
            <RawDataView data={PROTOTYPE_RAW_DATA} workspaceId="artifact-workspace-prototype" />
          </ExistingSurface>
        ) : null}

        {mappingVisible ? (
          <ExistingSurface
            title={`Measurement mapping · ${PROTOTYPE_NODE_META[selectedNode].label}`}
            artifact="measurement_structure"
            component="IndicatorTable"
            icon={Link2}
          >
            {selectedIndicators.length > 0 ? (
              <IndicatorTable indicators={selectedIndicators} />
            ) : (
              <NoDirectIndicator selectedNode={selectedNode} />
            )}
          </ExistingSurface>
        ) : null}

        {qualityVisible ? (
          <ExistingSurface
            title="Indicator health"
            artifact="validation_report"
            component="ValidationReportView"
            icon={BadgeCheck}
          >
            {selectedIndicators.length > 0 ? (
              <ValidationReportView data={selectedValidation} />
            ) : (
              <NoDirectIndicator selectedNode={selectedNode} />
            )}
          </ExistingSurface>
        ) : null}

        {observationsVisible ? (
          <ExistingSurface
            title="Canonical observations"
            artifact="measurements + panel"
            component="MeasurementsView"
            icon={Rows3}
            className="xl:col-span-2"
          >
            {selectedIndicators.length > 0 ? (
              <MeasurementsView
                data={selectedMeasurements}
                workspaceId="artifact-workspace-prototype"
              />
            ) : (
              <NoDirectIndicator selectedNode={selectedNode} />
            )}
          </ExistingSurface>
        ) : null}

        {fitVisible ? (
          <ExistingSurface
            title="Posterior predictive fit"
            artifact="posterior.ppc"
            component="PPCWarningsTable"
            icon={ChartSpline}
            className="xl:col-span-2"
          >
            {selectedPpcWarnings.length > 0 ? (
              <PPCWarningsTable
                warnings={selectedPpcWarnings}
                testStats={selectedPpcStats}
                overlays={selectedPpcOverlays}
              />
            ) : (
              <div className="rounded-lg border border-dashed bg-slate-50/60 p-5 text-center text-xs text-muted-foreground">
                No posterior-predictive slice is included for this construct in the compact
                prototype fixture.
              </div>
            )}
          </ExistingSurface>
        ) : null}

        {!sourceVisible &&
        !mappingVisible &&
        !observationsVisible &&
        !qualityVisible &&
        !fitVisible ? (
          <div className="grid min-h-56 place-items-center rounded-xl border border-dashed bg-white/60 text-xs text-muted-foreground xl:col-span-2">
            Turn on a materialized data layer to inspect it.
          </div>
        ) : null}
      </div>
    </div>
  );
}
