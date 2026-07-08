/**
 * analysis scenario model.
 *
 * Every analysis "scenario" is a materialized `simulate` tool result — a start
 * state + a list of timed latent clamps — sourced from the persisted output
 * trace (plus any injected extra message streams, e.g. dev mocks), carrying the
 * full per-node `visualization` trajectories that drive the living DAG. Two
 * provenances, distinguished purely by whether any clamp is applied:
 *
 *  - **baseline** — the no-intervention reference world (`clamps: []`): the system
 *    evolving under its own dynamics. There is at most one, shown first.
 *  - **intervention** — one or more `do()` clamps applied at some point.
 *
 * Each scenario also carries the natural-language **blurb** the LLM produced
 * alongside it (the assistant text co-located with the `simulate` tool call),
 * which explains the reasoning behind the intervention and its result.
 */

import type {
  LLMTrace,
  LatentStructureData,
  StatisticalModelSpecData,
  PosteriorData,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { formatScenarioActionDescription } from "@/components/dag/intervention-dag-semantics";
import type {
  EdgePosterior,
  AnalysisSimulationResult,
} from "@/components/dag/intervention-dag-types";
import { parseFixedEffect } from "@/lib/utils/ssm-latex";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";

// ── scenario types ──────────────────────────────────────────────────────────

export type ScenarioProvenance = "baseline" | "intervention";

/** Summary statistics for the rail card + effect summary. */
export interface ScenarioSummaryStats {
  mean: number;
  lower95: number;
  upper95: number;
  probPositive: number;
  peakEffect: number | null;
  timeToPeakDays: number | null;
}

export interface BaselineReportScenario {
  /** Stable selection key — the `simulate` tool-call id. */
  key: string;
  provenance: ScenarioProvenance;
  /** Concise label for the rail card. */
  title: string;
  outcome: string;
  summary: ScenarioSummaryStats;
  manifestEffects: Record<string, number> | null;
  result: AnalysisSimulationResult;
  requestedHorizonDays?: number;
  /** The user prompt that minted this scenario. */
  userQuery?: string;
  /** LLM-authored explanation produced with this scenario (assistant text beside the tool call). */
  blurb?: string;
}

// ── simulation sourcing ─────────────────────────────────────────────────────

const SIMULATION_TOOLS = new Set(["simulate"]);

function isSimulationResult(value: unknown): value is AnalysisSimulationResult {
  if (typeof value !== "object" || value == null) {
    return false;
  }
  const candidate = value as Partial<AnalysisSimulationResult> & { error?: unknown };
  if (candidate.error != null) {
    return false;
  }
  return (
    typeof candidate.outcome === "string" &&
    typeof candidate.summary === "object" &&
    candidate.summary != null &&
    Array.isArray(candidate.clamps) &&
    typeof candidate.start === "object" &&
    candidate.start != null
  );
}

/**
 * Tool outputs arrive as objects from the live refinement chat but as JSON
 * strings from a persisted trace (`TraceMessage.tool_result` is a string).
 * Coerce both to the structured result.
 */
function coerceSimOutput(output: unknown): AnalysisSimulationResult | null {
  if (typeof output === "string") {
    try {
      const parsed = JSON.parse(output) as unknown;
      return isSimulationResult(parsed) ? parsed : null;
    } catch {
      return null;
    }
  }
  return isSimulationResult(output) ? output : null;
}

function normalizeManifest(
  manifest: { [k: string]: number | undefined } | null | undefined,
): Record<string, number> | null {
  if (!manifest) {
    return null;
  }
  const entries = Object.entries(manifest).filter(
    (entry): entry is [string, number] => typeof entry[1] === "number",
  );
  return entries.length > 0 ? Object.fromEntries(entries) : null;
}

function readHorizonDays(input: unknown): number | undefined {
  if (typeof input !== "object" || input == null) {
    return undefined;
  }
  const query = (input as { query?: unknown }).query;
  if (typeof query !== "object" || query == null) {
    return undefined;
  }
  const horizon = (query as { horizon_days?: unknown }).horizon_days;
  return typeof horizon === "number" ? horizon : undefined;
}

function peakOfTrajectory(
  result: AnalysisSimulationResult,
): { day: number; effect: number } | null {
  const trajectory = result.effect_trajectory;
  if (!trajectory || trajectory.length === 0) {
    return null;
  }
  return trajectory.reduce(
    (best, point) => (Math.abs(point.effect) > Math.abs(best.effect) ? point : best),
    trajectory[0],
  );
}

interface RawSimulation {
  toolCallId: string;
  result: AnalysisSimulationResult;
  input: unknown;
  userQuery?: string;
  blurb?: string;
  order: number;
}

/**
 * Walk a UI-message stream and record every materialized `simulate` result,
 * keyed by tool-call id. The assistant text co-located with a simulate call is
 * captured as that scenario's `blurb`. `order` increases with recency; later
 * sources (and later occurrences) overwrite earlier ones with a higher order.
 */
function collectSimulations(
  messages: UIMessage[],
  into: Map<string, RawSimulation>,
  startOrder: number,
): number {
  let order = startOrder;
  let lastUserQuery: string | undefined;
  for (const message of messages) {
    if (message.role === "user") {
      const textPart = message.parts.find((part) => part.type === "text");
      if (textPart?.type === "text") {
        lastUserQuery = textPart.text;
      }
      continue;
    }
    if (message.role !== "assistant") {
      continue;
    }
    const blurb =
      message.parts
        .filter((part): part is Extract<typeof part, { type: "text" }> => part.type === "text")
        .map((part) => part.text)
        .join("\n\n")
        .trim() || undefined;
    for (const part of message.parts) {
      if (
        part.type !== "dynamic-tool" ||
        part.state !== "output-available" ||
        !SIMULATION_TOOLS.has(part.toolName)
      ) {
        continue;
      }
      const result = coerceSimOutput(part.output);
      if (!result) {
        continue;
      }
      into.set(part.toolCallId, {
        toolCallId: part.toolCallId,
        result,
        input: part.input,
        userQuery: lastUserQuery,
        blurb,
        order: order++,
      });
    }
  }
  return order;
}

function toScenario(raw: RawSimulation): BaselineReportScenario {
  const peak = peakOfTrajectory(raw.result);
  const isBaseline = raw.result.clamps.length === 0;
  return {
    key: raw.toolCallId,
    provenance: isBaseline ? "baseline" : "intervention",
    title: isBaseline ? "No intervention" : formatScenarioActionDescription(raw.result),
    outcome: raw.result.outcome,
    summary: {
      mean: raw.result.summary.mean,
      lower95: raw.result.summary.lower_95,
      upper95: raw.result.summary.upper_95,
      probPositive: raw.result.summary.prob_positive,
      peakEffect: peak ? peak.effect : null,
      timeToPeakDays: peak ? peak.day : null,
    },
    manifestEffects: normalizeManifest(raw.result.manifest_effects),
    result: raw.result,
    requestedHorizonDays: readHorizonDays(raw.input),
    userQuery: raw.userQuery,
    blurb: raw.blurb,
  };
}

/**
 * Build the ordered scenario list: the no-intervention baseline first (the most
 * recent clamp-less simulation, if any), then the intervention scenarios newest
 * first. The first entry is the sensible default selection.
 */
export function buildBaselineReportScenarios(args: {
  trace?: LLMTrace | null;
  /** Additional UI-message streams to source simulations from (e.g. dev mocks). */
  extraMessages?: UIMessage[];
}): BaselineReportScenario[] {
  const { trace, extraMessages } = args;

  const simulations = new Map<string, RawSimulation>();
  let order = 0;
  if (trace) {
    order = collectSimulations(traceToUIMessages(trace), simulations, order);
  }
  collectSimulations(extraMessages ?? [], simulations, order);

  const scenarios = [...simulations.values()]
    .sort((left, right) => right.order - left.order)
    .map(toScenario);

  const baseline = scenarios.find((scenario) => scenario.provenance === "baseline") ?? null;
  const interventions = scenarios.filter((scenario) => scenario.provenance === "intervention");

  return baseline ? [baseline, ...interventions] : interventions;
}

// ── edge posteriors (graph-level, scenario-independent) ──────────────────────

function normalizeConstructLabel(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[\s_]+/g, " ");
}

function resolveConstructName(label: string, constructNames: string[]): string | null {
  const normalizedLabel = normalizeConstructLabel(label);
  return (
    constructNames.find(
      (constructName) => normalizeConstructLabel(constructName) === normalizedLabel,
    ) ?? null
  );
}

function parseFixedEffectDescription(
  description: string,
  constructNames: string[],
): { source: string; target: string } | null {
  const match = /^Effect of (.+?) on (.+?)(?: \(|$)/.exec(description);
  if (!match) {
    return null;
  }
  const source = resolveConstructName(match[1], constructNames);
  const target = resolveConstructName(match[2], constructNames);
  if (!source || !target) {
    return null;
  }
  return { source, target };
}

/** Map fixed-effect posterior marginals onto `source→target` edge posteriors. */
export function buildEdgePosteriors({
  latentStructure,
  modelSpec,
  posterior,
}: {
  latentStructure?: LatentStructureData;
  modelSpec?: StatisticalModelSpecData;
  posterior?: PosteriorData;
}): Record<string, EdgePosterior> {
  if (!latentStructure) {
    return {};
  }

  const constructNames = latentStructure.latent_structure.constructs.map(
    (construct) => construct.name,
  );
  const parametersByName = new Map(
    (modelSpec?.statistical_model_spec.parameters ?? []).map((parameter) => [
      parameter.name,
      parameter,
    ]),
  );
  const marginals = posterior?.posterior_marginals ?? [];
  const edgePosteriors: Record<string, EdgePosterior> = {};

  for (const marginal of marginals) {
    if (!marginal.parameter.startsWith("beta_")) {
      continue;
    }
    const parameter = parametersByName.get(marginal.parameter);
    const parsed =
      (parameter?.description
        ? parseFixedEffectDescription(parameter.description, constructNames)
        : null) ?? parseFixedEffect(marginal.parameter, constructNames);
    if (!parsed) {
      continue;
    }
    edgePosteriors[`${parsed.source}→${parsed.target}`] = {
      mean: marginal.mean,
      ci_lower: marginal.hdi_3,
      ci_upper: marginal.hdi_97,
    };
  }

  return edgePosteriors;
}
