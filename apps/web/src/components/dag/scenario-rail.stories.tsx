import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { buildBaselineReportScenarios } from "@/components/pipeline/output-views/baseline-report-scenarios";
import { withContainer } from "@/components/story-decorators";
import { demoBaselineTrace } from "./__fixtures__/baseline_report-materialized-fixture";
import { ScenarioRail } from "./scenario-rail";

const scenarios = buildBaselineReportScenarios({ trace: demoBaselineTrace });

function RailDemo() {
  const [selected, setSelected] = useState<string | null>(scenarios[0]?.key ?? null);
  return <ScenarioRail scenarios={scenarios} selectedKey={selected} onSelect={setSelected} />;
}

const meta = {
  title: "Pipeline/Outputs/Baseline Report/Scenario Rail",
  component: ScenarioRail,
  decorators: [withContainer("max-w-4xl")],
} satisfies Meta<typeof ScenarioRail>;

export default meta;

export const Default: StoryObj = {
  render: () => <RailDemo />,
};
