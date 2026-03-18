import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage1bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage1bContent from "./stage-1b-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage = STAGES.find((s) => s.id === "stage-1b")!;
const data = fixture as Stage1bData;

const meta = {
  title: "Pipeline/Stages/1b – Measurement [gated]",
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

export const Pending: Story = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="pending"
      context={stage.description}
      hasGate
    />
  ),
};

export const Running: Story = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="running"
      context={stage.description}
      hasGate
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
      hasGate
      gateOverridden={data.gate_overridden ?? undefined}
      elapsedMs={18_900}
    >
      <Stage1bContent data={data} />
    </StageSection>
  ),
};

export const Failed: Story = {
  render: () => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="failed"
      context={stage.description}
      hasGate
    />
  ),
};
