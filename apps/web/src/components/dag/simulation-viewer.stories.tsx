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
import { SimulationViewer } from "./simulation-viewer";

/**
 * Stage 6 simulation viewer driven by the ideal materialized artifact.
 *
 * Two layers (per the stage's design):
 *  - Generative — the chat (right) mints new `simulate_*` scenarios. Disabled read-only.
 *  - Presentational — the viewer (left) re-slices a fixed scenario: pick a card,
 *    toggle the metric, scrub time. Selection is shared: chat "View" ↔ rail.
 *
 * Scenarios are sourced from the persisted trace via `buildStage6Scenarios`, so
 * this exercises the real production path (including string→object coercion).
 */

const scenarios = buildStage6Scenarios({
  interventionResults: materializedStage6Data.intervention_results,
  outcomeName,
  trace: materializedTrace,
});

const graph = { constructs, edges, indicators, edgePosteriors };

function SimulationViewerWithChat({ readOnly }: { readOnly: boolean }) {
  const [selectedKey, setSelectedKey] = useState<string | null>(scenarios[0]?.key ?? null);
  const [input, setInput] = useState("");

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(360px,1fr)]">
      <SimulationViewer
        scenarios={scenarios}
        graph={graph}
        finalSummary={materializedStage6Data.final_summary}
        selectedKey={selectedKey}
        onSelect={setSelectedKey}
        rankingResults={materializedStage6Data.intervention_results}
      />
      <div className="flex h-[760px] min-h-0 flex-col rounded-lg border bg-muted/30 p-3">
        <LLMTracePanelView
          trace={materializedTrace}
          canRefine={!readOnly}
          input={input}
          onInputChange={setInput}
          onSubmit={(event) => event.preventDefault()}
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
