import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { STAGES } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import {
  constructs,
  edgePosteriors,
  edges,
  indicators,
  materializedStage6Data,
  materializedTrace,
} from "@/components/dag/__fixtures__/stage-6-materialized-fixture";
import { SimulationViewer } from "@/components/dag/simulation-viewer";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import { createStageStatusStory, stageStoryDecorators } from "../stage-story-helpers";
import { StageStoryTemplate } from "../stage-story-template";
import { buildStage6Scenarios } from "./stage-6-scenarios";

const stage = STAGES.find((s) => s.id === "stage-6")!;
const graph = { constructs, edges, indicators, edgePosteriors };
const interventionResults = materializedStage6Data.intervention_results;

function scenariosFor(trace: LLMTrace | null) {
  return buildStage6Scenarios({ trace });
}

const withSimsScenarios = scenariosFor(materializedTrace);
const noSimsScenarios = scenariosFor(null);

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects/Panel",
  component: SimulationViewer,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof SimulationViewer>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Failed = createStageStatusStory(stage, "failed");

function CompletedInShell({ scenarios }: { scenarios: ReturnType<typeof buildStage6Scenarios> }) {
  const [selectedKey, setSelectedKey] = useState<string | null>(scenarios[0]?.key ?? null);

  return (
    <StageStoryTemplate
      stage={stage}
      status="completed"
      elapsedMs={9_400}
      trace={materializedTrace}
      panelContent={
        <LLMTracePanelView
          trace={materializedTrace}
          selectedSimulationKey={selectedKey ?? undefined}
          onSelectSimulation={(key) => setSelectedKey(key)}
        />
      }
    >
      <SimulationViewer
        scenarios={scenarios}
        graph={graph}
        selectedKey={selectedKey}
        onSelect={setSelectedKey}
        rankingResults={interventionResults}
      />
    </StageStoryTemplate>
  );
}

export const Completed: StoryObj = {
  name: "Completed (with simulations)",
  render: () => <CompletedInShell scenarios={withSimsScenarios} />,
};

export const CompletedNoSimulations: StoryObj = {
  name: "Completed (no simulations — ranking only)",
  render: () => <CompletedInShell scenarios={noSimsScenarios} />,
};
