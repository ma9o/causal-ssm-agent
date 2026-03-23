import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { LLMTrace, Stage6Data } from "@causal-ssm/api-types";
import { Badge } from "@/components/ui/badge";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import { TooltipProvider } from "@/components/ui/tooltip";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { Clock, Cpu } from "lucide-react";
import { StageSection } from "../stage-section";
import Stage6Content from "./stage-6-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-6.json";
import nutsdaFixture from "../../../../../../data/DOCTOLIB/run/stage-6-nutsda.json";

const stage = STAGES.find((s) => s.id === "stage-6")!;
const data = fixture as unknown as Stage6Data;
const nutsdaData = nutsdaFixture as unknown as Stage6Data;
const storyTrace: LLMTrace = {
  model: "openrouter/anthropic/claude-sonnet-4",
  total_time_seconds: 4.2,
  usage: {
    input_tokens: 1765,
    output_tokens: 312,
    reasoning_tokens: 94,
  },
  messages: [
    {
      role: "system",
      content:
        "You are writing the opening commentary for Stage 6 of a causal state-space analysis. Comment on the treatment-effect results for a technical user.",
      tool_is_error: false,
    },
    {
      role: "user",
      content:
        "Comment the results of Stage 6 for the fitted model, note warnings, and mention available rung 2 and rung 3 follow-up simulations.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "Statin adherence and blood-pressure medication adherence appear to be the strongest identifiable levers in the baseline ranking, both pointing toward lower downstream cardiovascular risk. The main caveat is that the fit still carries sensitivity and posterior-predictive warnings for some variables, so the ranking is informative but not fully clean. You can now inspect model details or ask for Pearl rung 2 and rung 3 simulations directly from this stage.",
      tool_is_error: false,
    },
  ],
};
const dataWithTrace = {
  ...data,
  llm_trace: storyTrace,
  final_summary:
    "Statin adherence and blood-pressure medication adherence appear to be the strongest identifiable levers in the baseline ranking, both pointing toward lower downstream cardiovascular risk. The main caveat is that the fit still carries sensitivity and posterior-predictive warnings for some variables, so the ranking is informative but not fully clean. You can now inspect model details or ask for Pearl rung 2 and rung 3 simulations directly from this stage.",
} as Stage6Data;

function StoryTracePanel({ trace }: { trace: LLMTrace }) {
  const messages = traceToUIMessages(trace);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="shrink-0 flex flex-wrap items-center gap-2 border-b bg-background pb-2 text-xs">
        <Badge variant="secondary" className="gap-1 text-[10px]">
          <Cpu className="h-3 w-3" />
          {trace.model}
        </Badge>
        <span className="text-muted-foreground">
          {formatCompact(trace.usage.input_tokens)} in / {formatCompact(trace.usage.output_tokens)} out
        </span>
        {trace.usage.reasoning_tokens ? (
          <span className="text-muted-foreground">
            ({formatCompact(trace.usage.reasoning_tokens)} reasoning)
          </span>
        ) : null}
        <span className="ml-auto flex items-center gap-1 text-muted-foreground">
          <Clock className="h-3 w-3" />
          {trace.total_time_seconds.toFixed(1)}s
        </span>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto">
        <ChatMessages messages={messages} />
      </div>
    </div>
  );
}

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects",
  component: Stage6Content,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-6xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof Stage6Content>;

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

export const CompletedSVI: Story = {
  name: "Completed (SVI / Laplace EM)",
  args: { data },
  render: (args) => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="completed"
      outcome={data.outcome}
      context={stage.description}
      elapsedMs={6_700}
    >
      <Stage6Content {...args} />
    </StageSection>
  ),
};

export const CompletedNUTS: Story = {
  name: "Completed (NUTS / DA)",
  args: { data: nutsdaData },
  render: (args) => (
    <StageSection
      number={stage.number}
      title={stage.label}
      status="completed"
      outcome={nutsdaData.outcome}
      context={stage.description}
      elapsedMs={8_100}
    >
      <Stage6Content {...args} />
    </StageSection>
  ),
};

export const CompletedWithTrace: Story = {
  name: "Completed With Trace",
  args: { data: dataWithTrace },
  render: (args) => (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
      <StageSection
        number={stage.number}
        title={stage.label}
        status="completed"
        outcome={dataWithTrace.outcome}
        context={stage.description}
        elapsedMs={6_700}
      >
        <Stage6Content {...args} />
      </StageSection>
      <div className="min-h-0 rounded-lg border bg-muted/30 p-3">
        <StoryTracePanel trace={storyTrace} />
      </div>
    </div>
  ),
};

export const Failed: StoryObj = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
