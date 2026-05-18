import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@nof1-causal-lab/api-types";
import type {
  LLMTrace,
  Stage1aData,
  Stage1bData,
  Stage4Data,
  Stage5bData,
  Stage6Data,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import { StoryStageLogView } from "../stage-story-log-stream";
import { StageStoryTemplate } from "../stage-story-template";
import Stage6Showcase from "./stage-6-showcase";
import { buildStage6DagScene } from "./stage-6-presentation";
import fixture from "../../../../../../data/DEMO_HEALTH/run/stage-6.json";
import auxGibbsFixture from "../../../../../../data/DEMO_HEALTH/run/stage-6-aux-gibbs.json";
import stage1aFixture from "../../../../../../data/DEMO_HEALTH/run/stage-1a.json";
import stage1bFixture from "../../../../../../data/DEMO_HEALTH/run/stage-1b.json";
import stage4Fixture from "../../../../../../data/DEMO_HEALTH/run/stage-4.json";
import stage5bFixture from "../../../../../../data/DEMO_HEALTH/run/stage-5b.json";
import stage5bAuxGibbsFixture from "../../../../../../data/DEMO_HEALTH/run/stage-5b-aux-gibbs.json";
import {
  counterfactualResult,
  interventionResult,
  mockTrace,
} from "@/components/dag/__fixtures__/intervention-dag-fixture";

type FollowUpScenario = "rung2" | "rung3";

const stage = STAGES.find((s) => s.id === "stage-6")!;
const stage1a = stage1aFixture as unknown as Stage1aData;
const stage1b = stage1bFixture as unknown as Stage1bData;
const stage4 = stage4Fixture as unknown as Stage4Data;
const stage5b = stage5bFixture as unknown as Stage5bData;
const stage5bAuxGibbs = stage5bAuxGibbsFixture as unknown as Stage5bData;
const data = { outcome: "success", ...fixture } as Stage6Data;
const auxGibbsData = { outcome: "success", ...auxGibbsFixture } as Stage6Data;
const storyTrace = mockTrace as LLMTrace;
const finalSummary =
  storyTrace.messages[storyTrace.messages.length - 1]?.content ??
  "Stage 6 baseline effects are available for follow-up simulations.";

const dataWithTrace = {
  ...data,
  llm_trace: storyTrace,
  final_summary: finalSummary,
} as Stage6Data;

const auxGibbsDataWithTrace = {
  ...auxGibbsData,
  llm_trace: storyTrace,
  final_summary: finalSummary,
} as Stage6Data;

const baselineDagScene = buildStage6DagScene({
  stage1a,
  stage1b,
  stage4,
  stage5b,
  refinementMessages: [],
  height: "600px",
});
const auxGibbsBaselineDagScene = buildStage6DagScene({
  stage1a,
  stage1b,
  stage4,
  stage5b: stage5bAuxGibbs,
  refinementMessages: [],
  height: "600px",
});

const completedShellProps = {
  outcome: dataWithTrace.outcome,
  elapsedMs: 6_700,
  trace: storyTrace,
};

const RUNG2_PROMPT = "What happens if we shift lipid burden by +1?";
const RUNG3_PROMPT = "What would have happened had medication adherence been higher?";

function getScenarioPrompt(scenario: FollowUpScenario): string {
  return scenario === "rung2" ? RUNG2_PROMPT : RUNG3_PROMPT;
}

function getSimulationResult(scenario: FollowUpScenario) {
  return scenario === "rung2" ? interventionResult : counterfactualResult;
}

function getResultHorizonDays(scenario: FollowUpScenario): number {
  const result = getSimulationResult(scenario);
  const lastDay = result.effect_trajectory?.[result.effect_trajectory.length - 1]?.day;
  return typeof lastDay === "number" && Number.isFinite(lastDay)
    ? Math.max(1, Math.round(lastDay))
    : 30;
}

function getToolName(scenario: FollowUpScenario): "simulate_intervention" | "simulate_counterfactual" {
  return scenario === "rung2" ? "simulate_intervention" : "simulate_counterfactual";
}

function getToolInput(scenario: FollowUpScenario) {
  if (scenario === "rung2") {
    return {
      action: interventionResult.action,
      outcome: interventionResult.outcome,
      query: {
        estimand: interventionResult.estimand,
        horizon_days: getResultHorizonDays(scenario),
        projection: "latent",
      },
    };
  }

  return {
    evidence: {
      start_time: counterfactualResult.evidence.start_time,
      end_time: counterfactualResult.evidence.end_time,
      variables: counterfactualResult.evidence.variables,
    },
    action: counterfactualResult.action,
    outcome: counterfactualResult.outcome,
    query: {
      estimand: counterfactualResult.estimand,
      horizon_days: getResultHorizonDays(scenario),
      projection: "latent",
    },
  };
}

function getAssistantSummary(scenario: FollowUpScenario): string {
  if (scenario === "rung2") {
    return "Rung 2 completed. Shifting lipid burden upward raises cardiovascular risk through vascular inflammation, with the strongest response accumulating over the forward trajectory.";
  }

  return "Rung 3 completed. Conditioning on the observed history and then increasing medication adherence lowers the projected cardiovascular-risk trajectory relative to the factual forecast.";
}

function createUserMessage(id: string, prompt: string): UIMessage {
  return {
    id,
    role: "user",
    parts: [{ type: "text", text: prompt }],
  };
}

function createAssistantMessage(id: string, scenario: FollowUpScenario): UIMessage {
  return {
    id,
    role: "assistant",
    parts: [
      { type: "text", text: getAssistantSummary(scenario) },
      {
        type: "dynamic-tool",
        toolCallId: `${id}-tool`,
        toolName: getToolName(scenario),
        state: "output-available",
        input: getToolInput(scenario),
        output: structuredClone(getSimulationResult(scenario)),
      },
    ],
  };
}

function createUnsupportedAssistantMessage(id: string): UIMessage {
  return {
    id,
    role: "assistant",
    parts: [
      {
        type: "text",
        text: "This story only supports two canned follow-ups: a rung 2 \"What happens if we shift lipid burden by +1?\" and a rung 3 \"What would have happened had medication adherence been higher?\"",
      },
    ],
  };
}

function matchScenario(prompt: string): FollowUpScenario | null {
  const normalized = prompt.trim().toLowerCase();

  if (
    normalized === RUNG2_PROMPT.toLowerCase() ||
    (normalized.includes("lipid") && normalized.includes("shift"))
  ) {
    return "rung2";
  }

  if (
    normalized === RUNG3_PROMPT.toLowerCase() ||
    (normalized.includes("medication") && normalized.includes("higher"))
  ) {
    return "rung3";
  }

  return null;
}

function buildFollowUpMessages(scenario: FollowUpScenario): UIMessage[] {
  return [
    createUserMessage(`follow-up-${scenario}-user`, getScenarioPrompt(scenario)),
    createAssistantMessage(`follow-up-${scenario}-assistant`, scenario),
  ];
}

function Stage6FollowUpPanel({
  refinementMessages,
  input = "",
  canRefine = false,
  isLoading = false,
  onInputChange,
  onSubmit,
}: {
  refinementMessages: UIMessage[];
  input?: string;
  canRefine?: boolean;
  isLoading?: boolean;
  onInputChange?: (value: string) => void;
  onSubmit?: (event: FormEvent) => void;
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <LLMTracePanelView
        trace={storyTrace}
        refinementMessages={refinementMessages}
        canRefine={canRefine}
        isLoading={isLoading}
        input={input}
        onInputChange={onInputChange}
        onSubmit={onSubmit}
      />
    </div>
  );
}

function InteractiveFollowUpDemoView() {
  const [refinementMessages, setRefinementMessages] = useState<UIMessage[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const timeoutRef = useRef<number | null>(null);
  const dagScene = buildStage6DagScene({
    stage1a,
    stage1b,
    stage4,
    stage5b,
    refinementMessages,
    height: "600px",
  });

  useEffect(() => {
    return () => {
      if (timeoutRef.current != null) {
        window.clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  function dispatchPrompt(prompt: string) {
    const trimmed = prompt.trim();
    if (!trimmed) return;

    if (timeoutRef.current != null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    const scenario = matchScenario(trimmed);
    const userMessage = createUserMessage(`interactive-${Date.now()}-user`, trimmed);

    if (!scenario) {
      setRefinementMessages((messages) => [
        ...messages,
        userMessage,
        createUnsupportedAssistantMessage(`interactive-${Date.now()}-unsupported`),
      ]);
      setInput("");
      setIsLoading(false);
      return;
    }

    setRefinementMessages((messages) => [...messages, userMessage]);
    setInput("");
    setIsLoading(true);

    timeoutRef.current = window.setTimeout(() => {
      setRefinementMessages((messages) => [
        ...messages,
        createAssistantMessage(`interactive-${Date.now()}-assistant`, scenario),
      ]);
      setIsLoading(false);
      timeoutRef.current = null;
    }, 450);
  }

  function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (isLoading) return;
    dispatchPrompt(input);
  }

  return (
    <StageStoryTemplate
      stage={stage}
      status="completed"
      outcome={dataWithTrace.outcome}
      elapsedMs={6_700}
      trace={storyTrace}
      defaultPanelOpen
      logView={
        <StoryStageLogView
          storyId="stage-6-interactive-follow-up-demo"
          status="completed"
        />
      }
      panelContent={
        <Stage6FollowUpPanel
          refinementMessages={refinementMessages}
          input={input}
          canRefine
          isLoading={isLoading}
          onInputChange={setInput}
          onSubmit={handleSubmit}
        />
      }
    >
      <Stage6Showcase
        data={dataWithTrace}
        dagScene={dagScene}
      />
    </StageStoryTemplate>
  );
}

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects",
  component: Stage6Showcase,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage6Showcase>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Completed = createCompletedStageStory({
  name: "Completed (MAP)",
  stage,
  args: {
    data: dataWithTrace,
    dagScene: baselineDagScene,
  },
  ...completedShellProps,
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const OpenPanel = createCompletedStageStory({
  stage,
  args: {
    data: dataWithTrace,
    dagScene: baselineDagScene,
  },
  ...completedShellProps,
  defaultPanelOpen: true,
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const CompletedAuxGibbs = createCompletedStageStory({
  name: "Completed (Aux Gibbs)",
  stage,
  args: {
    data: auxGibbsDataWithTrace,
    dagScene: auxGibbsBaselineDagScene,
  },
  outcome: auxGibbsDataWithTrace.outcome,
  elapsedMs: 8_100,
  trace: storyTrace,
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const Rung2FollowUp = createCompletedStageStory({
  name: "Completed (Rung 2 Follow-up)",
  stage,
  args: {
    data: dataWithTrace,
    dagScene: buildStage6DagScene({
      stage1a,
      stage1b,
      stage4,
      stage5b,
      refinementMessages: buildFollowUpMessages("rung2"),
      height: "600px",
    }),
  },
  ...completedShellProps,
  defaultPanelOpen: true,
  panelContent: (
    <Stage6FollowUpPanel refinementMessages={buildFollowUpMessages("rung2")} />
  ),
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const Rung3FollowUp = createCompletedStageStory({
  name: "Completed (Rung 3 Follow-up)",
  stage,
  args: {
    data: dataWithTrace,
    dagScene: buildStage6DagScene({
      stage1a,
      stage1b,
      stage4,
      stage5b,
      refinementMessages: buildFollowUpMessages("rung3"),
      height: "600px",
    }),
  },
  ...completedShellProps,
  defaultPanelOpen: true,
  panelContent: (
    <Stage6FollowUpPanel refinementMessages={buildFollowUpMessages("rung3")} />
  ),
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const InteractiveFollowUpDemo: StoryObj<typeof meta> = {
  name: "Interactive Follow-up Demo",
  parameters: {
    docs: {
      description: {
        story:
          "Use the follow-up chat input with either `What happens if we shift lipid burden by +1?` or `What would have happened had medication adherence been higher?`.",
      },
    },
  },
  render: () => <InteractiveFollowUpDemoView />,
};

export const Failed = createStageStatusStory(stage, "failed");
