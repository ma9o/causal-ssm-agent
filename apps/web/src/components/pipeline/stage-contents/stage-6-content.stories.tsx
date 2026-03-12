import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage6Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage6Content from "./stage-6-content";
import fixture from "../../../../../../packages/fixtures/doctolib/stage-6.json";
import nutsdaFixture from "../../../../../../packages/fixtures/doctolib/stage-6-nutsda.json";

const stage = STAGES.find((s) => s.id === "stage-6")!;
const data = fixture as Stage6Data;
const nutsdaData = nutsdaFixture as Stage6Data;

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects",
  component: Stage6Content,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage6Content>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Pending: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="pending" context={stage.description} />
  ),
};

export const Running: Story = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="running"
      context={stage.description}
      loadingHint={stage.loadingHint}
    />
  ),
};

export const CompletedSVI: Story = {
  name: "Completed (SVI / Laplace EM)",
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="completed"
      outcome={data.outcome}
      context={stage.description}
      elapsedMs={6_700}
    >
      <Stage6Content data={data} />
    </StageSection>
  ),
};

export const CompletedNUTS: Story = {
  name: "Completed (NUTS / DA)",
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="completed"
      outcome={nutsdaData.outcome}
      context={stage.description}
      elapsedMs={8_100}
    >
      <Stage6Content data={nutsdaData} />
    </StageSection>
  ),
};

export const Failed: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
