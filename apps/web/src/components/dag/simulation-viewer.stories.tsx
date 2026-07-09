import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { buildBaselineReportScenarios } from "@/components/pipeline/output-views/baseline-report-scenarios";
import { withContainer } from "@/components/story-decorators";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import {
  constructs,
  edgePosteriors,
  edges,
  indicators,
  materializedBaselineReportData,
  demoBaselineTrace,
  outcomeName,
} from "./__fixtures__/baseline_report-materialized-fixture";
import {
  buildDevMockMessages,
  makeMockSimulate,
  synthesizeMockScenarios,
} from "./interactive/dev-mock-scenario";
import { SimulationViewer } from "./simulation-viewer";

/**
 * analysis simulation viewer driven by the ideal materialized artifact.
 *
 * Two layers (per the output view design):
 *  - Generative — direct `simulate` dispatch mints new scenarios. Disabled read-only.
 *  - Presentational — the viewer (left) shows a rail of scenarios (the no-intervention
 *    baseline first, then interventions), the LLM's blurb for the focused scenario,
 *    and the living DAG. Selection is shared: chat "View" ↔ rail.
 *
 * Scenarios come from the persisted trace (exercising the real string→object
 * coercion + per-scenario blurbs) merged with synthesized rich mock scenarios
 * (full drift + bands + indicators) so the viewer demonstrates the complete
 * living DAG in context.
 */

const mockScenarios = synthesizeMockScenarios(constructs, edges, indicators, outcomeName);
const scenarios = buildBaselineReportScenarios({
  trace: demoBaselineTrace,
  extraMessages: buildDevMockMessages(mockScenarios),
});

const graph = { constructs, edges, indicators, edgePosteriors };
const mockSimulate = makeMockSimulate(mockScenarios.baseline.result);

function SimulationViewerWithChat({ readOnly }: { readOnly: boolean }) {
  const [selectedKey, setSelectedKey] = useState<string | null>(scenarios[0]?.key ?? null);

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(360px,1fr)]">
      <SimulationViewer
        scenarios={scenarios}
        graph={graph}
        selectedKey={selectedKey}
        onSelect={setSelectedKey}
        rankingResults={materializedBaselineReportData.intervention_results}
        onSimulate={readOnly ? undefined : mockSimulate}
      />
      <div className="flex h-[760px] min-h-0 flex-col rounded-lg border bg-muted/30 p-3">
        <LLMTracePanelView
          trace={demoBaselineTrace}
          selectedSimulationKey={selectedKey ?? undefined}
          onSelectSimulation={(key) => setSelectedKey(key)}
        />
      </div>
    </div>
  );
}

const meta = {
  title: "Pipeline/Outputs/Baseline Report/Simulation Viewer",
  component: SimulationViewer,
  decorators: [withContainer("max-w-6xl")],
} satisfies Meta<typeof SimulationViewer>;

export default meta;

export const LiveSession: StoryObj = {
  name: "Live session (read-write)",
  render: () => <SimulationViewerWithChat readOnly={false} />,
};

export const ReadOnly: StoryObj = {
  name: "Read-only (materialized)",
  render: () => <SimulationViewerWithChat readOnly={true} />,
};
