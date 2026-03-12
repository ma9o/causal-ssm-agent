import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage2Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage2Content from "./stage-2-content";
import fixture from "../../../../../../packages/fixtures/doctolib/stage-2.json";

const stage = STAGES.find((s) => s.id === "stage-2")!;
const data = fixture as Stage2Data;

const meta = {
  title: "Pipeline/Stages/2 – Data Extraction",
  component: Stage2Content,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage2Content>;

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
      elapsedMs={45_200}
    >
      <Stage2Content data={data} />
    </StageSection>
  ),
};

export const Failed: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
