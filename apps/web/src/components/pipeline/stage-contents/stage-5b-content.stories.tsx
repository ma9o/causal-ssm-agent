import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "../stage-section";
import Stage5bContent from "./stage-5b-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-5b.json";
import nutsdaFixture from "../../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";

const stage = STAGES.find((s) => s.id === "stage-5b")!;
const data = fixture as Stage5bData;
const nutsdaData = nutsdaFixture as Stage5bData;

const meta = {
  title: "Pipeline/Stages/5b – Inference & Diagnostics",
  component: Stage5bContent,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage5bContent>;

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
      elapsedMs={124_500}
    >
      <Stage5bContent data={data} />
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
      elapsedMs={342_000}
    >
      <Stage5bContent data={nutsdaData} />
    </StageSection>
  ),
};

export const Failed: Story = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
