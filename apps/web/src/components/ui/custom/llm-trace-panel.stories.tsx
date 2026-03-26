import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { LLMTrace } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Bot, X } from "lucide-react";
import { useState } from "react";
import { LLMTracePanelView } from "./llm-trace-panel-view";

const mockTrace: LLMTrace = {
  model: "claude-sonnet-4-20250514",
  total_time_seconds: 42.7,
  usage: {
    input_tokens: 4800,
    output_tokens: 8200,
    reasoning_tokens: 2150,
  },
  messages: [
    {
      role: "system",
      content:
        "You are a clinical epidemiologist specializing in cardiometabolic causal inference. Your task is to construct a latent causal model capturing the key factors driving the patient's cardiovascular risk.",
      tool_is_error: false,
    },
    {
      role: "user",
      content:
        "Research question: What factors drive this patient's cardiovascular risk, given 3 years of medical records including prescriptions, lab results, and vital signs?",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "I'll construct a latent causal model for this patient's cardiovascular risk trajectory. I'm including two unobserved confounders that are clinically important but not captured in routine records.\n\n**Key constructs identified:**\n\n| Construct | Role | Observable |\n|---|---|---|\n| Lipid Burden | Treatment-modifiable | Yes |\n| Vascular Inflammation | Mediator | Yes |\n| Glycemic Control | Treatment-modifiable | Yes |\n| Arterial Pressure | Treatment-modifiable | Yes |\n| Medication Adherence | Exposure | Yes |\n| Cardiovascular Risk | Outcome | Yes |\n| Genetic Predisposition | Confounder | No |\n| Lifestyle Factors | Confounder | No |",
      reasoning:
        "Patient has T2DM, hypertension, hypercholesterolemia. Key observed constructs: lipid burden (statin-modifiable), vascular inflammation (mediator of lipid→CV pathway), glycemic control (metformin-modifiable). Need to include unobserved confounders: genetic predisposition (affects both lipid burden and CV risk directly) and lifestyle factors (diet, exercise — affects glycemic control and arterial pressure).",
      tool_calls: [
        {
          id: "call_validate_01",
          name: "validate_latent_model",
          arguments: {
            structure_json: JSON.stringify({
              constructs: [
                { name: "lipid_burden", role: "endogenous" },
                { name: "vascular_inflammation", role: "endogenous" },
                { name: "cardiovascular_risk", role: "endogenous" },
              ],
              edges: [
                { from: "lipid_burden", to: "vascular_inflammation" },
                { from: "vascular_inflammation", to: "cardiovascular_risk" },
              ],
            }),
          },
        },
      ],
      tool_is_error: false,
    },
    {
      role: "tool",
      content: "VALID",
      tool_name: "validate_latent_model",
      tool_call_id: "call_validate_01",
      tool_result: "VALID",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "The latent model captures 8 constructs (6 observed, 2 unobserved) and 11 directed edges. The contemporaneous subgraph is acyclic.\n\nNotably, lipid_burden affects cardiovascular_risk only through vascular_inflammation, enabling frontdoor identification even in the presence of genetic confounding.",
      tool_is_error: false,
    },
    {
      role: "user",
      content:
        "Can you walk through the identifiability implications of each unobserved confounder?",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "Let me analyze each unobserved confounder's impact on identifiability.\n\n**1. Genetic Predisposition (U1)**\n\nThis confounder creates a backdoor path: `lipid_burden ← genetic_predisposition → cardiovascular_risk`. But lipid_burden has NO direct edge to cv_risk — it goes through vascular_inflammation only. This means frontdoor criteria may apply.\n\n**2. Lifestyle Factors (U2)**\n\nThis confounder affects both glycemic_control and arterial_pressure. Since these are both treatment-modifiable and we observe medication_adherence (which acts as an instrument), we can use instrumental variable estimation for the glycemic_control → cv_risk pathway.",
      reasoning:
        "For U1 (genetic_predisposition): Creates backdoor lipid_burden ← U1 → cv_risk. But lipid_burden has NO direct edge to cv_risk — it goes through vascular_inflammation only. This means frontdoor criteria may apply for the lipid_burden → vascular_inflammation → cv_risk pathway.\n\nFor U2 (lifestyle_factors): Creates backdoor glycemic_control ← U2 → arterial_pressure. But medication_adherence is an instrument — it affects glycemic_control directly but not U2.",
      tool_calls: [
        {
          id: "call_temporal_01",
          name: "check_temporal_identification",
          arguments: {
            dag_json: JSON.stringify({
              nodes: ["lipid_burden_t", "vasc_inflammation_t", "cv_risk_t"],
              lagged_edges: ["lipid_burden_t->lipid_burden_t1"],
            }),
          },
        },
      ],
      tool_is_error: false,
    },
    {
      role: "tool",
      content:
        "Temporal identification check passed. 4 of 5 causal effects are identifiable in the unrolled DAG. The glycemic_control → cardiovascular_risk effect remains partially identified (bounds width: 0.15).",
      tool_name: "check_temporal_identification",
      tool_call_id: "call_temporal_01",
      tool_result:
        "Temporal identification check passed. 4 of 5 causal effects are identifiable in the unrolled DAG.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "The temporal identification analysis confirms:\n\n| Causal Effect | Status | Method |\n|---|---|---|\n| lipid_burden → cardiovascular_risk | Identifiable | Frontdoor via vascular_inflammation |\n| vascular_inflammation → cardiovascular_risk | Identifiable | Direct (no confounders) |\n| arterial_pressure → cardiovascular_risk | Identifiable | IV via medication_adherence |\n| medication_adherence → glycemic_control | Identifiable | Direct (no confounders) |\n| glycemic_control → cardiovascular_risk | Partial | Bounds width 0.15 |",
      tool_is_error: false,
    },
  ],
};

function PanelShell({
  children,
  height = 600,
  interactive = false,
}: { children: React.ReactNode; height?: number; interactive?: boolean }) {
  return (
    <div
      className="mx-auto max-w-md overflow-hidden rounded-lg border bg-card shadow-sm"
      style={{ height }}
    >
      <div className="flex items-center justify-between border-b px-3 py-2.5">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Bot className="h-4 w-4 text-primary" />
          Assistant Details
        </div>
        <button
          type="button"
          className="rounded-md p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
      <div className="flex flex-col" style={{ height: "calc(100% - 45px)" }}>
        {children}
      </div>
    </div>
  );
}

function InteractivePanel({ trace }: { trace: LLMTrace }) {
  const [input, setInput] = useState("");
  return (
    <PanelShell height={700} interactive>
      <LLMTracePanelView
        trace={trace}
        canRefine
        input={input}
        onInputChange={setInput}
        onSubmit={(e) => e.preventDefault()}
      />
    </PanelShell>
  );
}

const meta = {
  title: "Pipeline/AssistantDetailsPanel",
  component: LLMTracePanelView,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof LLMTracePanelView>;

export default meta;
type Story = StoryObj<typeof meta>;

export const ReadOnly: Story = {
  args: { trace: mockTrace },
  render: (args) => (
    <PanelShell>
      <LLMTracePanelView {...args} />
    </PanelShell>
  ),
};

export const Interactive: Story = {
  args: { trace: mockTrace },
  render: (args) => <InteractivePanel trace={args.trace} />,
};

export const ShortTrace: Story = {
  args: {
    trace: {
      model: "claude-sonnet-4-20250514",
      total_time_seconds: 8.3,
      usage: { input_tokens: 1200, output_tokens: 2400, reasoning_tokens: null },
      messages: [
        {
          role: "user",
          content: "What is the primary outcome variable?",
          tool_is_error: false,
        },
        {
          role: "assistant",
          content:
            "Based on the research question and available data, the primary outcome variable is **cardiovascular risk**, measured through a composite of blood pressure readings, lipid panels, and HbA1c levels.",
          tool_is_error: false,
        },
      ],
    },
  },
  render: (args) => (
    <PanelShell height={400}>
      <LLMTracePanelView {...args} />
    </PanelShell>
  ),
};

export const WithToolError: Story = {
  args: {
    trace: {
      model: "claude-sonnet-4-20250514",
      total_time_seconds: 15.2,
      usage: { input_tokens: 2100, output_tokens: 3500, reasoning_tokens: 800 },
      messages: [
        {
          role: "user",
          content: "Validate this causal structure.",
          tool_is_error: false,
        },
        {
          role: "assistant",
          content: "Let me validate the proposed causal structure.",
          tool_calls: [
            {
              id: "call_err_01",
              name: "validate_latent_model",
              arguments: { structure_json: '{"constructs": [], "edges": []}' },
            },
          ],
          tool_is_error: false,
        },
        {
          role: "tool",
          content:
            "Error: DAG contains a cycle: A → B → C → A. Remove at least one edge to make the graph acyclic.",
          tool_name: "validate_latent_model",
          tool_call_id: "call_err_01",
          tool_result: "Error: DAG contains a cycle: A → B → C → A",
          tool_is_error: true,
        },
        {
          role: "assistant",
          content:
            "The validation found a cycle in the graph: `A → B → C → A`. I'll remove the `C → A` edge since the reverse direction (`A → C`) is more plausible given the temporal ordering.",
          tool_is_error: false,
        },
      ],
    },
  },
  render: (args) => (
    <PanelShell height={500}>
      <LLMTracePanelView {...args} />
    </PanelShell>
  ),
};
