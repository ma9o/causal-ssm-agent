import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage1bData, Stage2Data, Stage4Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage4Content from "./stage-4-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-4.json";
import stage2Fixture from "../../../../../../data/DOCTOLIB/run/stage-2.json";
import stage1bFixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage = STAGES.find((s) => s.id === "stage-4")!;
const data = fixture as unknown as Stage4Data;
const extractions = (stage2Fixture as unknown as Stage2Data).combined_extractions_sample;
const indicators = (stage1bFixture as unknown as Stage1bData).causal_spec.measurement.indicators;

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

export const Pending: StoryObj = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="pending" context={stage.description} />
  ),
};

export const Running: StoryObj = {
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
  args: { data, extractions, indicators },
  render: (args) => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="completed"
      outcome={data.outcome}
      context={stage.description}
      elapsedMs={15_600}
    >
      <Stage4Content {...args} />
    </StageSection>
  ),
};

export const Failed: StoryObj = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
