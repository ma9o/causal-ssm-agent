import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { StageOutcome } from "@causal-ssm/api-types";
import type { PrefectLogEntry } from "@/lib/prefect-log-client";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { withContainer } from "@/components/story-decorators";
import { useEffect, useState } from "react";
import { StageSection } from "./stage-section";
import { StageLogView } from "./stage-log-viewer";
import { createStageStoryLogs, makeStageStoryLog } from "./stage-story-log-fixture";

// ---------------------------------------------------------------------------
// Mock log data
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Helper: streaming log view that adds logs one by one
// ---------------------------------------------------------------------------

function StreamingLogView({ intervalMs = 600 }: { intervalMs?: number }) {
  const [logs, setLogs] = useState<PrefectLogEntry[]>([]);
  const storyLogs = createStageStoryLogs();

  useEffect(() => {
    let i = 0;
    const id = setInterval(() => {
      if (i >= storyLogs.length) {
        clearInterval(id);
        return;
      }
      setLogs((prev) => [...prev, makeStageStoryLog(i, storyLogs.length)]);
      i++;
    }, intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, storyLogs.length]);

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
  return (
    <StageLogView
      logs={createStageStoryLogs()}
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
  decorators: [withContainer("max-w-3xl")],
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
