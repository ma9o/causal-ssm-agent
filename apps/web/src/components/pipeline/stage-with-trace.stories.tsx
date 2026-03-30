import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { LLMTrace, Stage1aData } from "@causal-ssm/api-types";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import { TooltipProvider } from "@/components/ui/tooltip";
import { StageSection } from "./stage-section";
import Stage1aContent from "./stage-contents/stage-1a-content";
import { StageWithTraceView } from "./stage-with-trace";
import fixture from "../../../../../data/DOCTOLIB/run/stage-1a.json";

const stage = STAGES.find((s) => s.id === "stage-1a")!;
const data = fixture as unknown as Stage1aData & { llm_trace: LLMTrace };

const meta = {
  title: "Pipeline/StageWithTrace",
  component: StageWithTraceView,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="w-full p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof StageWithTraceView>;

export default meta;

export const Collapsed: StoryObj = {
  render: () => (
    <StageWithTraceView
      panelContent={<LLMTracePanelView trace={data.llm_trace} canRefine input="" />}
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
    </StageWithTraceView>
  ),
};

export const OpenWithRefinementInput: StoryObj = {
  render: () => (
    <StageWithTraceView
      panelContent={<LLMTracePanelView trace={data.llm_trace} canRefine input="" />}
      defaultOpen
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
    </StageWithTraceView>
  ),
};

export const OpenReadOnly: StoryObj = {
  render: () => (
    <StageWithTraceView
      panelContent={<LLMTracePanelView trace={data.llm_trace} />}
      defaultOpen
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
    </StageWithTraceView>
  ),
};
