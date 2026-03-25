import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage1bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage1bContent from "./stage-1b-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage = STAGES.find((s) => s.id === "stage-1b")!;
const data = fixture as unknown as Stage1bData;

const meta = {
  title: "Pipeline/Stages/1b – Measurement",
  component: Stage1bContent,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage1bContent>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Pending: StoryObj = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="pending"
      context={stage.description}
    />
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
      elapsedMs={18_900}
    >
      <Stage1bContent {...args} />
    </StageSection>
  ),
};

export const Failed: StoryObj = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="failed"
      context={stage.description}
    />
  ),
};
