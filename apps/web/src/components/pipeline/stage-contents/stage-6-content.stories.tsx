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
import type {
  RefinementUIMessage,
  SuggestionAction,
  SuggestionChip,
} from "@/lib/utils/trace-to-core";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import { StoryStageLogView } from "../stage-story-log-stream";
import { StageStoryTemplate } from "../stage-story-template";
import Stage6Showcase from "./stage-6-showcase";
import { buildStage6DagScene } from "./stage-6-presentation";
import fixture from "../../../../../../data/DEMO/run/stage-6.json";
import auxKalmanMCMCFixture from "../../../../../../data/DEMO/run/stage-6.json";
import stage1aFixture from "../../../../../../data/DEMO/run/stage-1a.json";
import stage1bFixture from "../../../../../../data/DEMO/run/stage-1b.json";
import stage4Fixture from "../../../../../../data/DEMO/run/stage-4.json";
import stage5bFixture from "../../../../../../data/DEMO/run/stage-5b.json";
import stage5bAuxKalmanMCMCFixture from "../../../../../../data/DEMO/run/stage-5b.json";
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
const stage5bAuxKalmanMCMC = stage5bAuxKalmanMCMCFixture as unknown as Stage5bData;
const data = fixture as Stage6Data;
const auxKalmanMCMCData = auxKalmanMCMCFixture as Stage6Data;
const storyTrace = mockTrace as LLMTrace;
const finalSummary =
  storyTrace.messages[storyTrace.messages.length - 1]?.content ??
  "Stage 6 baseline effects are available for follow-up simulations.";

const dataWithTrace = {
  ...data,
  llm_trace: storyTrace,
  final_summary: finalSummary,
} as Stage6Data;

const auxKalmanMCMCDataWithTrace = {
  ...auxKalmanMCMCData,
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
const auxKalmanMCMCBaselineDagScene = buildStage6DagScene({
  stage1a,
  stage1b,
  stage4,
  stage5b: stage5bAuxKalmanMCMC,
  refinementMessages: [],
  height: "600px",
});

const completedShellProps = {
  outcome: dataWithTrace.outcome,
  elapsedMs: 6_700,
  trace: storyTrace,
};

const RUNG2_PROMPT = "What happens if we shift serotonergic exposure by +1?";
const RUNG3_PROMPT = "What would have happened had adherence been higher?";

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

function getToolName(
  scenario: FollowUpScenario,
): "simulate_intervention" | "simulate_counterfactual" {
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
    start: counterfactualResult.start,
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
    return "Rung 2 completed. Shifting serotonergic exposure upward improves affective state, with downstream movement through sleep quality and physical activity.";
  }

  return "Rung 3 completed. Starting from the retained fitted state and then increasing adherence improves the projected affective-state trajectory relative to the factual forecast.";
}

function createUserMessage(id: string, prompt: string): UIMessage {
  return {
    id,
    role: "user",
    parts: [{ type: "text", text: prompt }],
  };
}

function getFollowUpSuggestions(scenario: FollowUpScenario): SuggestionChip[] {
  if (scenario === "rung2") {
    return [
      {
        label: "Counterfactual: higher adherence",
        action: {
          tool: "simulate_counterfactual",
          input: {
            start: counterfactualResult.start,
            action: counterfactualResult.action,
            outcome: counterfactualResult.outcome,
            query: {
              estimand: counterfactualResult.estimand,
              horizon_days: 30,
              projection: "latent",
            },
          },
        },
      },
      {
        label: "Run the same with a −1 shift",
        action: {
          tool: "simulate_intervention",
          input: {
            action: { ...interventionResult.action, amount: -1 },
            outcome: interventionResult.outcome,
            query: {
              estimand: interventionResult.estimand,
              horizon_days: 30,
              projection: "latent",
            },
          },
        },
      },
    ];
  }
  return [
    {
      label: "Try shifting exposure +1",
      action: {
        tool: "simulate_intervention",
        input: {
          action: interventionResult.action,
          outcome: interventionResult.outcome,
          query: {
            estimand: interventionResult.estimand,
            horizon_days: 30,
            projection: "latent",
          },
        },
      },
    },
  ];
}

function createAssistantMessage(id: string, scenario: FollowUpScenario): RefinementUIMessage {
  const suggestions = getFollowUpSuggestions(scenario);
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
      {
        type: "data-suggestions",
        data: { suggestions },
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
        text: 'This story only supports two canned follow-ups: a rung 2 "What happens if we shift serotonergic exposure by +1?" and a rung 3 "What would have happened had adherence been higher?"',
      },
    ],
  };
}

function matchScenario(prompt: string): FollowUpScenario | null {
  const normalized = prompt.trim().toLowerCase();

  if (
    normalized === RUNG2_PROMPT.toLowerCase() ||
    (normalized.includes("serotonergic") && normalized.includes("shift"))
  ) {
    return "rung2";
  }

  if (
    normalized === RUNG3_PROMPT.toLowerCase() ||
    (normalized.includes("adherence") && normalized.includes("higher"))
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
  onSuggestionClick,
}: {
  refinementMessages: UIMessage[];
  input?: string;
  canRefine?: boolean;
  isLoading?: boolean;
  onInputChange?: (value: string) => void;
  onSubmit?: (event: FormEvent) => void;
  onSuggestionClick?: (action: SuggestionAction, chip: SuggestionChip) => void;
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
        onSuggestionClick={onSuggestionClick}
      />
    </div>
  );
}

function InteractiveFollowUpDemoView() {
  const [refinementMessages, setRefinementMessages] = useState<UIMessage[]>(() =>
    buildFollowUpMessages("rung2"),
  );
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

  function dispatchAction(action: SuggestionAction, chip: SuggestionChip) {
    if (timeoutRef.current != null) {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }

    const scenario: FollowUpScenario =
      action.tool === "simulate_counterfactual" ? "rung3" : "rung2";
    const toolCallId = `chip-${Date.now()}`;
    const userMessage = createUserMessage(
      `chip-user-${toolCallId}`,
      `[Action] ${chip.label}`,
    );
    const pendingMessage: UIMessage = {
      id: `chip-assistant-${toolCallId}`,
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolCallId,
          toolName: action.tool,
          state: "input-available",
          input: action.input,
        },
      ],
    };

    setRefinementMessages((messages) => [...messages, userMessage, pendingMessage]);
    setIsLoading(true);

    timeoutRef.current = window.setTimeout(() => {
      setRefinementMessages((messages) =>
        messages.map((msg) => {
          if (msg.id !== pendingMessage.id) return msg;
          return {
            ...msg,
            parts: msg.parts.map((part) => {
              if (
                part.type !== "dynamic-tool" ||
                part.toolCallId !== toolCallId
              ) {
                return part;
              }
              return {
                type: "dynamic-tool" as const,
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                state: "output-available" as const,
                input: part.input,
                output: structuredClone(getSimulationResult(scenario)),
              };
            }),
          };
        }),
      );
      setIsLoading(false);
      timeoutRef.current = null;
    }, 450);
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
        <StoryStageLogView storyId="stage-6-interactive-follow-up-demo" status="completed" />
      }
      panelContent={
        <Stage6FollowUpPanel
          refinementMessages={refinementMessages}
          input={input}
          canRefine
          isLoading={isLoading}
          onInputChange={setInput}
          onSubmit={handleSubmit}
          onSuggestionClick={dispatchAction}
        />
      }
    >
      <Stage6Showcase data={dataWithTrace} dagScene={dagScene} />
    </StageStoryTemplate>
  );
}

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects/Panel",
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
  defaultPanelOpen: true,
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const CompletedAuxKalmanMCMC = createCompletedStageStory({
  name: "Completed (Auxiliary Kalman MCMC)",
  stage,
  args: {
    data: auxKalmanMCMCDataWithTrace,
    dagScene: auxKalmanMCMCBaselineDagScene,
  },
  outcome: auxKalmanMCMCDataWithTrace.outcome,
  elapsedMs: 8_100,
  trace: storyTrace,
  defaultPanelOpen: true,
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
  panelContent: <Stage6FollowUpPanel refinementMessages={buildFollowUpMessages("rung2")} />,
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
  panelContent: <Stage6FollowUpPanel refinementMessages={buildFollowUpMessages("rung3")} />,
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const InteractiveFollowUpDemo: StoryObj<typeof meta> = {
  name: "Interactive Follow-up Demo",
  args: {
    data: dataWithTrace,
    dagScene: baselineDagScene,
  },
  parameters: {
    docs: {
      description: {
        story:
          "Use the follow-up chat input with either `What happens if we shift serotonergic exposure by +1?` or `What would have happened had adherence been higher?`.",
      },
    },
  },
  render: () => <InteractiveFollowUpDemoView />,
};

export const Failed = createStageStatusStory(stage, "failed");
