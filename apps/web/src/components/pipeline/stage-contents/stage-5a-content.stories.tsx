import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage5aData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage5aContent from "./stage-5a-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-5a.json";

const stage = STAGES.find((s) => s.id === "stage-5a")!;
const data = fixture as Stage5aData;

const meta = {
  title: "Pipeline/Stages/5a – SVI Preflight",
  component: Stage5aContent,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage5aContent>;

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
      elapsedMs={32_100}
    >
      <Stage5aContent data={data} />
    </StageSection>
  ),
};

export const Failed: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
