import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { UIMessage } from "ai";
import { type ReactNode, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { withContainer } from "@/components/story-decorators";
import { LLMTracePanelView } from "@/components/ui/custom/llm-trace-panel-view";
import { cn } from "@/lib/utils";
import { formatNumber } from "@/lib/utils/format";
import { InterventionDag } from "./intervention-dag";
import type { Stage6SimulationResult } from "./intervention-dag-types";
import {
  counterfactualResult,
  edgePosteriors,
  interventionResult,
  mockTrace,
} from "./__fixtures__/intervention-dag-fixture";
import { constructs, edges, indicators } from "./__fixtures__/dag-base-fixtures";

/**
 * Design surface for the Stage 6 simulation viewer.
 *
 * The action space splits into two layers:
 *  - Generative (the chat): change *what was done* — intervention, conditioning,
 *    horizon → a new persisted scenario. Disabled in read-only.
 *  - Presentational (this viewer): change *how you look* at a fixed result —
 *    scrub time, focus a node, toggle metric/space, select a scenario. Always
 *    available, even read-only.
 *
 * Scenarios are persisted as `simulate_*` tool results on the assistant turn
 * that produced them. The chat is the real `LLMTracePanelView`; selection rides
 * the shared `ChatMessages` "View" affordance and drives the DAG. The left-hand
 * QueryHeader / ViewControls / EffectSummary are new viewer pieces that lift out
 * into a real component once the UX settles.
 */

interface Scenario {
  id: string;
  userQuery: string;
  answer: string;
  toolName: "simulate_intervention" | "simulate_counterfactual";
  input: Record<string, unknown>;
  result: Stage6SimulationResult;
}

const HORIZON_DAYS = 60;

const serotonergicScenario: Scenario = {
  id: "serotonergic-boost",
  userQuery: "What happens to affective state if we raise serotonergic exposure by 1 SD?",
  answer:
    "Raising serotonergic exposure lifts affective state by ~+0.31 SD at steady state, propagating through sleep quality and physical activity over roughly three weeks.",
  toolName: "simulate_intervention",
  input: {
    action: { variable: "serotonergic_exposure", mode: "shift", amount: 1.0 },
    outcome: "affective_state",
    query: { estimand: "trajectory", horizon_days: HORIZON_DAYS, projection: "latent" },
  },
  result: interventionResult,
};

const adherenceScenario: Scenario = {
  id: "adherence-cf",
  userQuery:
    "Counterfactually, if adherence had been 0.5 SD higher from late February, how would affective state have differed?",
  answer:
    "Conditioning on the fitted state at 2026-02-28, a +0.5 SD adherence shift improves the affective-state path by ~+0.24 SD relative to the factual forecast.",
  toolName: "simulate_counterfactual",
  input: {
    start: { time: "2026-02-28T00:00:00+00:00" },
    action: { variable: "adherence", mode: "shift", amount: 0.5 },
    outcome: "affective_state",
    query: { estimand: "trajectory", horizon_days: HORIZON_DAYS, projection: "latent" },
  },
  result: counterfactualResult,
};

const SCENARIOS: Scenario[] = [serotonergicScenario, adherenceScenario];

// Build the conversation the way the real refinement chat produces it: a user
// turn, then an assistant turn carrying a `simulate_*` tool result.
function userMessage(scenario: Scenario): UIMessage {
  return {
    id: `${scenario.id}-user`,
    role: "user",
    parts: [{ type: "text", text: scenario.userQuery }],
  };
}

function assistantMessage(scenario: Scenario): UIMessage {
  return {
    id: `${scenario.id}-assistant`,
    role: "assistant",
    parts: [
      { type: "text", text: scenario.answer },
      {
        type: "dynamic-tool",
        toolCallId: scenario.id,
        toolName: scenario.toolName,
        state: "output-available",
        input: scenario.input,
        output: scenario.result,
      },
    ],
  };
}

const MESSAGES: UIMessage[] = SCENARIOS.flatMap((scenario) => [
  userMessage(scenario),
  assistantMessage(scenario),
]);

// ── prototype derivations (precomputed in the contract eventually) ────

function describeAction(result: Stage6SimulationResult): string {
  const { action } = result;
  if (action.mode === "set") {
    return `do(${action.variable} := ${formatNumber(action.value ?? 0, 2)})`;
  }
  const amount = action.amount ?? 0;
  return `do(${action.variable} ${amount >= 0 ? "+=" : "-="} ${formatNumber(Math.abs(amount), 2)} SD)`;
}

function peakOf(result: Stage6SimulationResult): { day: number; effect: number } {
  return (result.effect_trajectory ?? []).reduce(
    (best, point) => (Math.abs(point.effect) > Math.abs(best.effect) ? point : best),
    { day: 0, effect: 0 },
  );
}

function signed(value: number): string {
  return `${value >= 0 ? "+" : ""}${formatNumber(value)}`;
}

// ── read-only query header: what defines this simulation ──────────────

function MetaChip({ children }: { children: ReactNode }) {
  return (
    <span className="inline-flex items-center rounded-md border bg-background px-2 py-0.5 font-mono text-[11px] text-muted-foreground">
      {children}
    </span>
  );
}

function QueryHeader({ result }: { result: Stage6SimulationResult }) {
  return (
    <div className="space-y-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant={result.rung === 2 ? "secondary" : "outline"}>
          {result.rung === 2 ? "Interventional · rung 2" : "Counterfactual · rung 3"}
        </Badge>
        <MetaChip>{describeAction(result)}</MetaChip>
        <MetaChip>→ {result.outcome}</MetaChip>
        <MetaChip>{result.estimand}</MetaChip>
        <MetaChip>{HORIZON_DAYS}-day</MetaChip>
        {result.rung === 3 ? <MetaChip>from {result.start.time?.slice(0, 10)}</MetaChip> : null}
      </div>
      <p className="text-[11px] text-muted-foreground">
        These define the simulation — change them by asking in chat; each answer mints a new
        scenario.
      </p>
    </div>
  );
}

// ── presentational view controls (re-slice a fixed result) ────────────

function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (next: T) => void;
  options: { value: T; label: string }[];
}) {
  return (
    <div className="inline-flex rounded-md border p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={cn(
            "rounded px-2 py-0.5 text-[11px] font-medium transition-colors",
            value === option.value
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ViewControls() {
  const [metric, setMetric] = useState<"effect" | "action" | "reference">("effect");
  const [space, setSpace] = useState<"latent" | "manifest">("latent");
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] text-muted-foreground">Show</span>
        <Segmented
          value={metric}
          onChange={setMetric}
          options={[
            { value: "effect", label: "Effect Δ" },
            { value: "action", label: "Action path" },
            { value: "reference", label: "Reference" },
          ]}
        />
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] text-muted-foreground">Space</span>
        <Segmented
          value={space}
          onChange={setSpace}
          options={[
            { value: "latent", label: "Latent" },
            { value: "manifest", label: "Manifest" },
          ]}
        />
      </div>
      <span className="text-[11px] text-muted-foreground/70">
        presentational — wires into the DAG view-model next
      </span>
    </div>
  );
}

// ── effect summary for the focused outcome ────────────────────────────

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="space-y-0.5">
      <div className="text-[11px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="font-mono text-sm">{value}</div>
      {hint ? <div className="text-[10px] text-muted-foreground">{hint}</div> : null}
    </div>
  );
}

function EffectSummary({ result }: { result: Stage6SimulationResult }) {
  const { summary } = result;
  const peak = peakOf(result);
  const reference =
    result.rung === 2 ? result.baseline_treatment_mean : result.baseline_forecast_mean;
  return (
    <div className="grid grid-cols-2 gap-4 rounded-lg border bg-muted/20 p-3 sm:grid-cols-4">
      <Stat
        label={`Effect on ${result.outcome}`}
        value={`${signed(summary.mean)} SD`}
        hint={`95% CI [${formatNumber(summary.lower_95)}, ${formatNumber(summary.upper_95)}]`}
      />
      <Stat label="P(effect > 0)" value={`${Math.round(summary.prob_positive * 100)}%`} />
      <Stat label="Peak effect" value={`${signed(peak.effect)} SD`} hint={`day ${peak.day}`} />
      <Stat label="Reference mean" value={formatNumber(reference)} />
    </div>
  );
}

// ── composition ───────────────────────────────────────────────────────

function SimulationViewer({ readOnly }: { readOnly: boolean }) {
  const [selectedId, setSelectedId] = useState(serotonergicScenario.id);
  const [input, setInput] = useState("");
  const selected = SCENARIOS.find((scenario) => scenario.id === selectedId) ?? serotonergicScenario;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,2fr)_minmax(360px,1fr)]">
      <div className="space-y-3">
        <QueryHeader result={selected.result} />
        <ViewControls />
        <InterventionDag
          constructs={constructs}
          edges={edges}
          indicators={indicators}
          edgePosteriors={edgePosteriors}
          requestedHorizonDays={HORIZON_DAYS}
          simulationResult={selected.result}
          height="560px"
        />
        <EffectSummary result={selected.result} />
      </div>
      <div className="flex h-[720px] min-h-0 flex-col rounded-lg border bg-muted/30 p-3">
        <LLMTracePanelView
          trace={mockTrace}
          refinementMessages={MESSAGES}
          canRefine={!readOnly}
          input={input}
          onInputChange={setInput}
          onSubmit={(event) => event.preventDefault()}
          selectedSimulationKey={selectedId}
          onSelectSimulation={(key) => setSelectedId(key)}
        />
      </div>
    </div>
  );
}

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects/Simulation Viewer",
  component: InterventionDag,
  decorators: [withContainer("max-w-6xl")],
} satisfies Meta<typeof InterventionDag>;

export default meta;

export const LiveSession: StoryObj = {
  name: "Live session (read-write)",
  render: () => <SimulationViewer readOnly={false} />,
};

export const ReadOnlyDemo: StoryObj = {
  name: "Read-only (DEMO, prepopulated)",
  render: () => <SimulationViewer readOnly={true} />,
};
