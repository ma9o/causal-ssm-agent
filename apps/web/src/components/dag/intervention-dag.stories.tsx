import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { LLMTrace } from "@causal-ssm/api-types";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { withContainer } from "@/components/story-decorators";
import { ChatMessages } from "@/components/ui/custom/chat-messages";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { Clock, Cpu } from "lucide-react";
import { useState } from "react";
import { InterventionDag } from "./intervention-dag";
import {
  counterfactualResult,
  edgePosteriors,
  interventionResult,
  mockTrace,
  processNoise,
} from "./__fixtures__/intervention-dag-fixture";
import { constructs, edges, indicators } from "./__fixtures__/dag-base-fixtures";

// ── Shared trace panel ────────────────────────────────────────────────

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
          {formatCompact(trace.usage.input_tokens)} in /{" "}
          {formatCompact(trace.usage.output_tokens)} out
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

// ── Story 1: Static weighted DAG ──────────────────────────────────────

function StaticDagWithControls() {
  const [showNoise, setShowNoise] = useState(false);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Switch
          checked={showNoise}
          onCheckedChange={(checked: boolean) => setShowNoise(checked)}
          size="sm"
        />
        <span className="text-xs text-muted-foreground">
          Show process noise (σ²)
        </span>
      </div>
      <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
        <InterventionDag
          constructs={constructs}
          edges={edges}
          indicators={indicators}
          edgePosteriors={edgePosteriors}
          processNoise={processNoise}
          showNoiseNodes={showNoise}
        />
        <div className="min-h-0 rounded-lg border bg-muted/30 p-3 h-[600px]">
          <StoryTracePanel trace={mockTrace} />
        </div>
      </div>
    </div>
  );
}

// ── Meta ──────────────────────────────────────────────────────────────

const meta = {
  title: "DAG/InterventionDag",
  component: InterventionDag,
  decorators: [withContainer("max-w-6xl")],
} satisfies Meta<typeof InterventionDag>;

export default meta;

// ── Stories ───────────────────────────────────────────────────────────

export const PosteriorWeighted: StoryObj = {
  name: "Posterior-Weighted DAG",
  render: () => <StaticDagWithControls />,
};

export const Rung2Intervention: StoryObj = {
  name: "Rung 2 \u2014 do(lipid_burden)",
  render: () => (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
      <InterventionDag
        constructs={constructs}
        edges={edges}
        indicators={indicators}
        edgePosteriors={edgePosteriors}
        processNoise={processNoise}
        simulationResult={interventionResult}
        height="600px"
      />
      <div className="min-h-0 rounded-lg border bg-muted/30 p-3 h-[600px]">
        <StoryTracePanel trace={mockTrace} />
      </div>
    </div>
  ),
};

export const Rung3Counterfactual: StoryObj = {
  name: "Rung 3 \u2014 Counterfactual (medication_adherence)",
  render: () => (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(320px,1fr)]">
      <InterventionDag
        constructs={constructs}
        edges={edges}
        indicators={indicators}
        edgePosteriors={edgePosteriors}
        processNoise={processNoise}
        simulationResult={counterfactualResult}
        height="600px"
      />
      <div className="min-h-0 rounded-lg border bg-muted/30 p-3 h-[600px]">
        <StoryTracePanel trace={mockTrace} />
      </div>
    </div>
  ),
};
