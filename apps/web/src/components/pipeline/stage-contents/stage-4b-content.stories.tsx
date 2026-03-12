import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage4bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage4bContent from "./stage-4b-content";
import fixture from "../../../../../../packages/fixtures/doctolib/stage-4b.json";

const stage = STAGES.find((s) => s.id === "stage-4b")!;
const data = fixture as Stage4bData;

const meta = {
  title: "Pipeline/Stages/4b – Parametric ID [gated]",
  component: Stage4bContent,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage4bContent>;

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
      elapsedMs={8_400}
    >
      <Stage4bContent data={data} />
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
