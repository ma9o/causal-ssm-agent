import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import {
  constructs,
  edgePosteriors,
  edges,
  indicators,
  materializedBaselineReportData,
  materializedTrace,
} from "@/components/dag/__fixtures__/baseline_report-materialized-fixture";
import { SimulationViewer } from "@/components/dag/simulation-viewer";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import { createOutputStatusStory, outputStoryDecorators } from "../output-story-helpers";
import { OutputStoryTemplate } from "../output-story-template";
import { buildBaselineReportScenarios } from "./baseline-report-scenarios";

const output = TRANSITIONS.find((s) => s.id === "baseline_report")!;
const graph = { constructs, edges, indicators, edgePosteriors };
const interventionResults = materializedBaselineReportData.intervention_results;

function scenariosFor(trace: LLMTrace | null) {
  return buildBaselineReportScenarios({ trace });
}

const withSimsScenarios = scenariosFor(materializedTrace);
const noSimsScenarios = scenariosFor(null);

const meta = {
  title: "Pipeline/Outputs/Baseline Report/Panel",
  component: SimulationViewer,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof SimulationViewer>;

export default meta;

export const Pending = createOutputStatusStory(output, "pending");

export const Running = createOutputStatusStory(output, "running");

export const Failed = createOutputStatusStory(output, "failed");

function CompletedInShell({
  scenarios,
}: {
  scenarios: ReturnType<typeof buildBaselineReportScenarios>;
}) {
  const [selectedKey, setSelectedKey] = useState<string | null>(scenarios[0]?.key ?? null);

  return (
    <OutputStoryTemplate
      output={output}
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
    </OutputStoryTemplate>
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
