import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage1bData, Stage2Data, Stage4Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage4Content from "./stage-4-content";
import fixture from "../../../../../../packages/fixtures/doctolib/stage-4.json";
import stage2Fixture from "../../../../../../packages/fixtures/doctolib/stage-2.json";
import stage1bFixture from "../../../../../../packages/fixtures/doctolib/stage-1b.json";

const stage = STAGES.find((s) => s.id === "stage-4")!;
const data = fixture as Stage4Data;
const extractions = (stage2Fixture as Stage2Data).combined_extractions_sample;
const indicators = (stage1bFixture as Stage1bData).causal_spec.measurement.indicators;

const meta = {
  title: "Pipeline/Stages/4 – Model Specification",
  component: Stage4Content,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage4Content>;

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

export const Completed: Story = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="completed"
      outcome={data.outcome}
      context={stage.description}
      elapsedMs={15_600}
    >
      <Stage4Content data={data} extractions={extractions} indicators={indicators} />
    </StageSection>
  ),
};

export const Failed: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
