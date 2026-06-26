import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { buildStage6Scenarios } from "@/components/pipeline/stage-contents/stage-6-scenarios";
import { withContainer } from "@/components/story-decorators";
import { materializedTrace } from "./__fixtures__/stage-6-materialized-fixture";
import { ScenarioRail } from "./scenario-rail";

const scenarios = buildStage6Scenarios({ trace: materializedTrace });

function RailDemo() {
  const [selected, setSelected] = useState<string | null>(scenarios[0]?.key ?? null);
  return <ScenarioRail scenarios={scenarios} selectedKey={selected} onSelect={setSelected} />;
}

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects/Scenario Rail",
  component: ScenarioRail,
  decorators: [withContainer("max-w-4xl")],
} satisfies Meta<typeof ScenarioRail>;

export default meta;

export const Default: StoryObj = {
  render: () => <RailDemo />,
};
