import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage1aData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage1aContent from "./stage-1a-content";
import fixture from "../../../../../../packages/fixtures/doctolib/stage-1a.json";

const stage = STAGES.find((s) => s.id === "stage-1a")!;
const data = fixture as Stage1aData;

const meta = {
  title: "Pipeline/Stages/1a – Latent Model",
  component: Stage1aContent,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage1aContent>;

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
      elapsedMs={12_450}
    >
      <Stage1aContent data={data} />
    </StageSection>
  ),
};

export const Failed: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
