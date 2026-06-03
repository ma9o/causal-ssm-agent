/**
 * Stage 6 scenario model.
 *
 * Unifies the two provenances of a Stage 6 "scenario" into a single selectable
 * set for the simulation viewer:
 *
 *  - **baseline** — one per `do(treatment += 1 SD) → outcome` entry in the Stage 6
 *    `intervention_results` ranking. Carries posterior draws + a coarse temporal
 *    decomposition + a manifest projection, but no per-node trajectories.
 *  - **simulation** — one per materialized `simulate` tool result (a start state +
 *    a list of timed latent clamps), sourced from the persisted trace ∪ the
 *    in-memory refinement conversation. Carries full per-node `visualization`
 *    trajectories that drive the animated DAG + trajectory chart.
 *
 * Replaces the latest-only `buildStage6DagScene` / `extractLatestStage6FollowUpSimulation`.
 */

import type {
  LLMTrace,
  Stage1aData,
  Stage4Data,
  Stage5bData,
  TreatmentEffect,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { formatScenarioActionDescription } from "@/components/dag/intervention-dag-semantics";
import type {
  EdgePosterior,
  Stage6SimulationResult,
} from "@/components/dag/intervention-dag-types";
import { parseFixedEffect } from "@/lib/utils/ssm-latex";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { drawsCI, meanDraws, probPositive } from "@/lib/utils/treatment-effect-stats";

/** Temporal decomposition carried by a baseline {@link TreatmentEffect}. */
type BaselineTemporal = NonNullable<TreatmentEffect["temporal"]>;

// ── scenario types ──────────────────────────────────────────────────────────

export type ScenarioProvenance = "baseline" | "simulation";

/** Summary statistics shared by both provenances, for the rail card + header. */
export interface ScenarioSummaryStats {
  mean: number;
  lower95: number;
  upper95: number;
  probPositive: number;
  peakEffect: number | null;
  timeToPeakDays: number | null;
}

interface ScenarioBase {
  /** Stable selection key. Simulations use the `toolCallId`; baselines use `baseline:<treatment>`. */
  key: string;
  provenance: ScenarioProvenance;
  /** Concise label for the rail card. */
  title: string;
  outcome: string;
  summary: ScenarioSummaryStats;
  manifestEffects: Record<string, number> | null;
}

export interface SimulationScenario extends ScenarioBase {
  provenance: "simulation";
  result: Stage6SimulationResult;
  requestedHorizonDays?: number;
  userQuery?: string;
}

export interface BaselineScenario extends ScenarioBase {
  provenance: "baseline";
  treatment: string;
  posteriorDraws: number[] | null;
  temporal: BaselineTemporal | null;
}

export type Stage6Scenario = SimulationScenario | BaselineScenario;

// ── simulation sourcing (trace ∪ refinement) ────────────────────────────────

const SIMULATION_TOOLS = new Set(["simulate"]);

function isSimulationResult(value: unknown): value is Stage6SimulationResult {
  if (typeof value !== "object" || value == null) {
    return false;
  }
  const candidate = value as Partial<Stage6SimulationResult> & { error?: unknown };
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
function coerceSimOutput(output: unknown): Stage6SimulationResult | null {
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

function peakOfTrajectory(result: Stage6SimulationResult): { day: number; effect: number } | null {
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
  result: Stage6SimulationResult;
  input: unknown;
  userQuery?: string;
  order: number;
}

/**
 * Walk a UI-message stream and record every materialized `simulate_*` result,
 * keyed by tool-call id. `order` increases with recency; later sources (and
 * later occurrences) overwrite earlier ones with a higher order.
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
        order: order++,
      });
    }
  }
  return order;
}

function toSimulationScenario(raw: RawSimulation): SimulationScenario {
  const peak = peakOfTrajectory(raw.result);
  return {
    key: raw.toolCallId,
    provenance: "simulation",
    title: formatScenarioActionDescription(raw.result),
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
  };
}

function toBaselineScenario(effect: TreatmentEffect, outcomeName: string): BaselineScenario | null {
  const draws = effect.posterior_draws ?? null;
  const mean = meanDraws(draws);
  const ci = drawsCI(draws);
  if (mean == null || ci == null) {
    return null;
  }
  return {
    key: `baseline:${effect.treatment}`,
    provenance: "baseline",
    title: effect.treatment,
    outcome: outcomeName,
    summary: {
      mean,
      lower95: ci.lower,
      upper95: ci.upper,
      probPositive: probPositive(draws) ?? 0,
      peakEffect: effect.temporal?.peak_effect ?? null,
      timeToPeakDays: effect.temporal?.time_to_peak_days ?? null,
    },
    manifestEffects: normalizeManifest(effect.manifest_effects),
    treatment: effect.treatment,
    posteriorDraws: draws,
    temporal: effect.temporal ?? null,
  };
}

/**
 * Build the unified, ordered scenario list: simulations (newest first), then
 * baseline treatments (by descending |effect|). The first entry is the sensible
 * default selection.
 */
export function buildStage6Scenarios(args: {
  interventionResults: TreatmentEffect[];
  outcomeName: string | null;
  trace?: LLMTrace | null;
  refinementMessages?: UIMessage[];
}): Stage6Scenario[] {
  const { interventionResults, outcomeName, trace, refinementMessages } = args;

  const simulations = new Map<string, RawSimulation>();
  let order = 0;
  if (trace) {
    order = collectSimulations(traceToUIMessages(trace), simulations, order);
  }
  collectSimulations(refinementMessages ?? [], simulations, order);

  const simulationScenarios = [...simulations.values()]
    .sort((left, right) => right.order - left.order)
    .map(toSimulationScenario);

  const baselineScenarios = interventionResults
    .map((effect) => toBaselineScenario(effect, outcomeName ?? "outcome"))
    .filter((scenario): scenario is BaselineScenario => scenario !== null)
    .sort((left, right) => Math.abs(right.summary.mean) - Math.abs(left.summary.mean));

  return [...simulationScenarios, ...baselineScenarios];
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
  stage1a,
  stage4,
  stage5b,
}: {
  stage1a?: Stage1aData;
  stage4?: Stage4Data;
  stage5b?: Stage5bData;
}): Record<string, EdgePosterior> {
  if (!stage1a) {
    return {};
  }

  const constructNames = stage1a.latent_model.constructs.map((construct) => construct.name);
  const parametersByName = new Map(
    (stage4?.model_spec.parameters ?? []).map((parameter) => [parameter.name, parameter]),
  );
  const marginals = stage5b?.posterior_marginals ?? [];
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
