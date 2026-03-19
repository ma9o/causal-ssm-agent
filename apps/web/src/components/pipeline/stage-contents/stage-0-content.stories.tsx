import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage0Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage0Content from "./stage-0-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-0.json";

const stage = STAGES.find((s) => s.id === "stage-0")!;
const data = fixture as Stage0Data;

const meta = {
  title: "Pipeline/Stages/0 – Preprocess",
  component: Stage0Content,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage0Content>;

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
  args: { data },
  render: (args) => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="completed"
      outcome={data.outcome}
      context={stage.description}
      elapsedMs={4_320}
    >
      <Stage0Content {...args} />
    </StageSection>
  ),
};

export const Failed: StoryObj = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
