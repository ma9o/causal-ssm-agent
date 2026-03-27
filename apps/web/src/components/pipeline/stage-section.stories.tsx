import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { StageOutcome } from "@causal-ssm/api-types";
import type { PrefectLogEntry } from "@/lib/prefect-log-client";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { useEffect, useState } from "react";
import { StageSection } from "./stage-section";
import { StageLogView } from "./stage-log-viewer";

// ---------------------------------------------------------------------------
// Mock log data
// ---------------------------------------------------------------------------

const MOCK_LOG_MESSAGES = [
  { level: 20, message: "Starting stage execution…" },
  { level: 20, message: "Loading dataset from workspace artifacts" },
  { level: 20, message: "Found 1,247 records across 3 data sources" },
  { level: 20, message: "Validating column schemas against causal specification" },
  { level: 30, message: "Column 'mood_score' has 12% missing values — will impute with LOCF" },
  { level: 20, message: "Encoding categorical variables (one-hot)" },
  { level: 20, message: "Submitting LLM request for causal structure proposal" },
  { level: 10, message: "POST /v1/chat/completions — model=gpt-4o, tokens_est=4,320" },
  { level: 20, message: "LLM response received (3.2s, 1,847 completion tokens)" },
  { level: 20, message: "Parsing DAG from structured output" },
  { level: 20, message: "DAG has 8 nodes, 11 edges — checking acyclicity" },
  { level: 20, message: "DAG is acyclic" },
  { level: 20, message: "Running d-separation tests against observed correlations" },
  { level: 30, message: "Implied independence Sleep→Appetite not supported (p=0.003) — flagging" },
  { level: 20, message: "Checking identification via ID algorithm (y0)" },
  { level: 10, message: "Projecting DAG to ADMG for identification check" },
  { level: 20, message: "Causal effect X→Y is identifiable via back-door adjustment" },
  { level: 20, message: "Computing propensity scores for confounders {Z1, Z2}" },
  { level: 40, message: "Propensity score model convergence warning: max_iter reached" },
  { level: 20, message: "Retrying with increased max_iter=500" },
  { level: 20, message: "Propensity model converged (iter=342)" },
  { level: 20, message: "Estimating ATE via inverse-probability weighting" },
  { level: 20, message: "ATE = 0.34 (95% CI: [0.12, 0.56])" },
  { level: 20, message: "Writing stage artifacts to workspace storage" },
  { level: 20, message: "Stage completed successfully" },
] satisfies Pick<PrefectLogEntry, "level" | "message">[];

function makeMockLog(index: number): PrefectLogEntry {
  const entry = MOCK_LOG_MESSAGES[index % MOCK_LOG_MESSAGES.length];
  const ts = new Date(Date.now() - (MOCK_LOG_MESSAGES.length - index) * 800);
  return {
    id: `mock-${index}`,
    created: ts.toISOString(),
    name: "prefect.flow_runs",
    level: entry.level,
    message: entry.message,
    timestamp: ts.toISOString(),
    flow_run_id: "mock-flow-run",
    task_run_id: null,
  };
}

// ---------------------------------------------------------------------------
// Helper: streaming log view that adds logs one by one
// ---------------------------------------------------------------------------

function StreamingLogView({ intervalMs = 600 }: { intervalMs?: number }) {
  const [logs, setLogs] = useState<PrefectLogEntry[]>([]);

  useEffect(() => {
    let i = 0;
    const id = setInterval(() => {
      if (i >= MOCK_LOG_MESSAGES.length) {
        clearInterval(id);
        return;
      }
      setLogs((prev) => [...prev, makeMockLog(i)]);
      i++;
    }, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs]);

  return (
    <StageLogView
      logs={logs}
      status="running"
      bootstrapStatus="success"
      connectionState="streaming"
    />
  );
}

function CompletedLogView() {
  const logs = MOCK_LOG_MESSAGES.map((_, i) => makeMockLog(i));
  return (
    <StageLogView
      logs={logs}
      status="completed"
      bootstrapStatus="success"
      connectionState="idle"
    />
  );
}

// ---------------------------------------------------------------------------
// Stories
// ---------------------------------------------------------------------------

const meta = {
  title: "Pipeline/StageSection",
  component: StageSection,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
  args: {
    number: "0",
    title: "Example Stage",
    status: "pending",
    context: "A stage description for demonstration purposes.",
  },
  argTypes: {
    status: {
      control: "select",
      options: ["pending", "running", "completed", "failed"] satisfies StageRunStatus[],
    },
    outcome: {
      control: "select",
      options: ["success", "warn", "fail"] satisfies StageOutcome[],
    },
  },
} satisfies Meta<typeof StageSection>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Playground: Story = {
  args: {
    number: "0",
    title: "Preprocess",
    status: "completed",
    outcome: "success",
    context: "Parses raw data files and prepares them for downstream analysis.",
    elapsedMs: 4_320,
    children: (
      <div className="rounded-md border bg-muted/30 p-4 text-sm text-muted-foreground">
        Stage content would appear here.
      </div>
    ),
  },
};

/** Running stage — logs stream inline in the card body. */
export const RunningWithStreamingLogs: Story = {
  args: {
    number: "1a",
    title: "Causal Specification",
    status: "running",
    context: "Generating causal specification from user question.",
    loadingHint: "Querying the LLM for causal structure…",
    logView: <StreamingLogView />,
  },
};

/** Completed stage — logs are behind a collapsible toggle. */
export const CompletedWithCollapsibleLogs: Story = {
  args: {
    number: "1a",
    title: "Causal Specification",
    status: "completed",
    outcome: "success",
    context: "Generates the causal specification from the user question.",
    elapsedMs: 12_500,
    logView: <CompletedLogView />,
    children: (
      <div className="rounded-md border bg-muted/30 p-4 text-sm text-muted-foreground">
        Completed stage content would appear here.
      </div>
    ),
  },
};
