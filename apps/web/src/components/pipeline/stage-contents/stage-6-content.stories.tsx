import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage6Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { UIMessage } from "ai";
import { StageSection } from "../stage-section";
import Stage6Content from "./stage-6-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-6.json";
import nutsdaFixture from "../../../../../../data/DOCTOLIB/run/stage-6-nutsda.json";

const stage = STAGES.find((s) => s.id === "stage-6")!;
const data = fixture as unknown as Stage6Data;
const nutsdaData = nutsdaFixture as unknown as Stage6Data;
const stage6AssistantMessages: UIMessage[] = [
  {
    id: "assistant-user-1",
    role: "user",
    parts: [{ type: "text", text: "Which treatments are identifiable and what looks strongest?" }],
  },
  {
    id: "assistant-reply-1",
    role: "assistant",
    parts: [
      {
        type: "text",
        text: "The fitted model currently supports read-only inspection plus rung 2 and rung 3 simulations. Statin adherence and antihypertensive adherence look like the strongest identifiable treatments in the baseline ranking.",
      },
      {
        type: "dynamic-tool",
        toolCallId: "tool-get-model-info",
        toolName: "get_model_info",
        state: "output-available",
        input: {
          sections: ["overview", "identifiability", "baseline_effects", "capabilities"],
        },
        output: {
          overview: {
            outcome: "CardiovascularRisk",
            treatments: ["StatinAdherence", "BPMedAdherence", "Exercise"],
          },
          identifiability: {
            identifiable_treatments: ["StatinAdherence", "BPMedAdherence", "Exercise"],
          },
          baseline_effects: [
            { treatment: "StatinAdherence", effect_size: -0.41, prob_positive: 0.02 },
            { treatment: "BPMedAdherence", effect_size: -0.28, prob_positive: 0.09 },
            { treatment: "Exercise", effect_size: -0.17, prob_positive: 0.14 },
          ],
          capabilities: {
            intervention: { rung: 2, estimands: ["steady_state", "trajectory"] },
            counterfactual: { rung: 3, estimands: ["end_state", "trajectory"] },
          },
        },
      },
    ],
  },
  {
    id: "assistant-user-2",
    role: "user",
    parts: [
      {
        type: "text",
        text: "Run a 30-day rung 2 trajectory where StatinAdherence is shifted up by 1 latent unit.",
      },
    ],
  },
  {
    id: "assistant-reply-2",
    role: "assistant",
    parts: [
      {
        type: "text",
        text: "Rung 2 suggests a sustained reduction in cardiovascular risk over the 30-day horizon, with most of the effect accumulating in the first week and then flattening toward a lower-risk steady state.",
      },
      {
        type: "dynamic-tool",
        toolCallId: "tool-rung-2",
        toolName: "simulate_intervention",
        state: "output-available",
        input: {
          action: { variable: "StatinAdherence", mode: "shift", amount: 1 },
          query: { estimand: "trajectory", horizon_days: 30, projection: "both" },
        },
        output: {
          rung: 2,
          outcome: "CardiovascularRisk",
          summary: {
            mean: -0.43,
            lower_95: -0.67,
            upper_95: -0.19,
            prob_positive: 0.01,
          },
          temporal: {
            effect_1d: -0.12,
            effect_7d: -0.31,
            effect_30d: -0.43,
            peak_effect: -0.45,
          },
          warnings: ["Prior sensitivity warning present for StatinAdherence."],
        },
      },
    ],
  },
  {
    id: "assistant-user-3",
    role: "user",
    parts: [
      {
        type: "text",
        text: "Now do rung 3 using the last two observed weeks as evidence and compare the final forecast.",
      },
    ],
  },
  {
    id: "assistant-reply-3",
    role: "assistant",
    parts: [
      {
        type: "text",
        text: "Conditioning on the recent observed history preserves the same direction but shrinks the magnitude slightly. The posterior state was conditioned with an approximate smoother step before applying the action.",
      },
      {
        type: "dynamic-tool",
        toolCallId: "tool-rung-3",
        toolName: "simulate_counterfactual",
        state: "output-available",
        input: {
          evidence: {
            mode: "observed_window",
            start_time: "2024-09-16T00:00:00Z",
            end_time: "2024-09-30T00:00:00Z",
          },
          action: { variable: "StatinAdherence", mode: "shift", amount: 1 },
          query: { estimand: "end_state", horizon_days: 30, projection: "latent" },
        },
        output: {
          rung: 3,
          baseline_forecast_mean: 1.34,
          counterfactual_forecast_mean: 0.98,
          summary: {
            mean: -0.36,
            lower_95: -0.58,
            upper_95: -0.11,
            prob_positive: 0.03,
          },
          warnings: [
            "Kalman smoother unavailable; counterfactual state estimated from the final observed measurement slice.",
          ],
        },
      },
    ],
  },
];

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

export const CompletedWithAssistant: Story = {
  name: "Completed With Assistant",
  args: {
    data,
    userId: "storybook-demo",
    assistantDemoState: {
      messages: stage6AssistantMessages,
      status: "ready",
      showExamplePrompts: true,
    },
  },
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

export const Failed: StoryObj = {
  render: () => (
    <StageSection number={stage.number} title={stage.label} status="failed" context={stage.description} />
  ),
};
