import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { LLMTracePanelView } from "./llm-trace-panel-view";

const minimalTrace: LLMTrace = {
  model: "anthropic/claude-sonnet-4-20250514",
  total_time_seconds: 12.3,
  usage: { input_tokens: 4200, output_tokens: 1800 },
  messages: [
    { role: "system", content: "You are a causal modeling assistant.", tool_is_error: false },
    {
      role: "user",
      content:
        "Identify the latent constructs and causal edges for a study on how sleep quality affects cognitive performance, controlling for stress levels.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "## Latent Constructs\n\n| Construct | Type | Description |\n|---|---|---|\n| SleepQuality | continuous | Self-reported and actigraphy-derived sleep quality |\n| CognitivePerformance | continuous | Composite of working memory and reaction time |\n| StressLevel | continuous | Perceived stress scale score |\n\n## Causal Edges\n\n- **SleepQuality → CognitivePerformance**: Poor sleep degrades attentional resources\n- **StressLevel → SleepQuality**: Stress disrupts sleep onset and continuity\n- **StressLevel → CognitivePerformance**: Acute stress impairs executive function",
      tool_is_error: false,
    },
  ],
};

const traceWithReasoning: LLMTrace = {
  model: "anthropic/claude-sonnet-4-20250514",
  total_time_seconds: 24.7,
  usage: { input_tokens: 8500, output_tokens: 3200, reasoning_tokens: 1500 },
  messages: [
    { role: "system", content: "You are a causal modeling assistant.", tool_is_error: false },
    {
      role: "user",
      content: "Check identifiability of the effect of X on Y given confounders Z1, Z2.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content: "Let me work through the identification analysis.",
      reasoning:
        "I need to check the back-door criterion first. The confounders Z1 and Z2 block all back-door paths from X to Y if there are no unobserved common causes. Let me enumerate the paths:\n\n1. X ← Z1 → Y (blocked by conditioning on Z1)\n2. X ← Z2 → Y (blocked by conditioning on Z2)\n3. X ← Z1 ← U → Z2 → Y (need to check if U is observed)\n\nSince U is unobserved, conditioning on Z1 and Z2 may open a collider path. I need to verify with the ID algorithm.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "## Identifiability Result\n\nThe causal effect `P(Y | do(X))` **is identifiable** via the back-door adjustment:\n\n```\nP(Y | do(X)) = Σ_{z1,z2} P(Y | X, z1, z2) P(z1, z2)\n```\n\nThe adjustment set `{Z1, Z2}` satisfies the back-door criterion because:\n- Both Z1 and Z2 are non-descendants of X\n- Together they block all back-door paths from X to Y",
      tool_is_error: false,
    },
  ],
};

const traceWithToolCalls: LLMTrace = {
  model: "anthropic/claude-sonnet-4-20250514",
  total_time_seconds: 18.5,
  usage: { input_tokens: 6100, output_tokens: 2400 },
  messages: [
    {
      role: "user",
      content: "Run the parametric identifiability check on the sleep-cognition model.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content: "I'll run the structural identifiability check on your model.",
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: {
            name: "check_parametric_id",
            arguments: JSON.stringify({
              model_name: "sleep_cognition_ssm",
              method: "transfer_function",
            }),
          },
        },
      ],
      tool_is_error: false,
    },
    {
      role: "tool",
      content: JSON.stringify({
        status: "identified",
        globally_identified: ["beta_sleep_cog", "sigma_stress"],
        locally_identified: ["alpha_stress_sleep"],
        unidentified: [],
      }),
      tool_call_id: "call_1",
      tool_name: "check_parametric_id",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "## Parametric Identifiability\n\nAll parameters are identified:\n\n| Parameter | Status |\n|---|---|\n| `beta_sleep_cog` | Globally identified |\n| `sigma_stress` | Globally identified |\n| `alpha_stress_sleep` | Locally identified |\n\nNo unidentified parameters — the model is structurally identifiable.",
      tool_is_error: false,
    },
  ],
};

const meta = {
  title: "UI/LLMTracePanelView",
  component: LLMTracePanelView,
  parameters: {
    layout: "padded",
  },
  decorators: [
    (Story) => (
      <div className="h-[600px] w-full max-w-2xl rounded-lg border bg-muted/30 p-3 flex flex-col">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof LLMTracePanelView>;

export default meta;
type Story = StoryObj<typeof meta>;

export const ReadOnly: Story = {
  args: { trace: minimalTrace },
};

export const WithReasoning: Story = {
  args: { trace: traceWithReasoning },
};

export const WithToolCalls: Story = {
  args: { trace: traceWithToolCalls },
};

export const WithRefinementInput: Story = {
  args: {
    trace: minimalTrace,
    canRefine: true,
    input: "",
  },
};

export const Loading: Story = {
  args: {
    trace: minimalTrace,
    canRefine: true,
    isLoading: true,
    input: "Add a mediator between stress and cognition",
  },
};
