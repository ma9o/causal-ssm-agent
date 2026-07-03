import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { buildStage6Scenarios } from "@/components/pipeline/stage-contents/stage-6-scenarios";
import { withContainer } from "@/components/story-decorators";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import {
  constructs,
  edgePosteriors,
  edges,
  indicators,
  materializedStage6Data,
  materializedTrace,
  outcomeName,
} from "./__fixtures__/stage-6-materialized-fixture";
import {
  buildDevMockMessages,
  makeMockSimulate,
  synthesizeMockScenarios,
} from "./interactive/dev-mock-scenario";
import { SimulationViewer } from "./simulation-viewer";

/**
 * Stage 6 simulation viewer driven by the ideal materialized artifact.
 *
 * Two layers (per the stage's design):
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
const scenarios = buildStage6Scenarios({
  trace: materializedTrace,
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
        rankingResults={materializedStage6Data.intervention_results}
        onSimulate={readOnly ? undefined : mockSimulate}
      />
      <div className="flex h-[760px] min-h-0 flex-col rounded-lg border bg-muted/30 p-3">
        <LLMTracePanelView
          trace={materializedTrace}
          selectedSimulationKey={selectedKey ?? undefined}
          onSelectSimulation={(key) => setSelectedKey(key)}
        />
      </div>
    </div>
  );
}

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects/Simulation Viewer",
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
