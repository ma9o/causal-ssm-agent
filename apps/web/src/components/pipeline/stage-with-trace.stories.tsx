import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage1aData, LLMTrace } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { RefinementProvider } from "@/lib/contexts/refinement-context";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import { StageSection } from "./stage-section";
import { StageWithTrace } from "./stage-with-trace";
import Stage1aContent from "./stage-contents/stage-1a-content";
import fixture from "../../../../../data/DOCTOLIB/run/stage-1a.json";

const stage = STAGES.find((s) => s.id === "stage-1a")!;
const data = fixture as unknown as Stage1aData & { llm_trace: LLMTrace };

const meta = {
  title: "Pipeline/StageWithTrace",
  component: StageWithTrace,
  decorators: [
    (Story) => (
      <RefinementProvider>
        <TooltipProvider>
          <div className="w-full p-4">
            <Story />
          </div>
        </TooltipProvider>
      </RefinementProvider>
    ),
  ],
} satisfies Meta<typeof StageWithTrace>;

export default meta;

export const Collapsed: StoryObj = {
  render: () => (
    <StageWithTrace
      panelContent={
        <LLMTracePanelView trace={data.llm_trace} canRefine input="" />
      }
    >
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
    </StageWithTrace>
  ),
};

export const WithRefinementInput: StoryObj = {
  render: () => (
    <StageWithTrace
      panelContent={
        <LLMTracePanelView trace={data.llm_trace} canRefine input="" />
      }
    >
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
    </StageWithTrace>
  ),
};

export const ReadOnly: StoryObj = {
  render: () => (
    <StageWithTrace
      panelContent={<LLMTracePanelView trace={data.llm_trace} />}
    >
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
    </StageWithTrace>
  ),
};
