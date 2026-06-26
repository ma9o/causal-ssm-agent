/**
 * DEV-ONLY mock for the interactive DAG. While the backend `simulate` tool isn't
 * in the local loop (and real stage-6 traces carry no `simulate` calls), this
 * synthesizes the per-node trajectories / drift / indicator visuals so the living
 * DAG is visible and editable on the live stage-6 page.
 *
 * The scenario *set* is data-driven: each `saved_scenarios` entry in the stage-6
 * artifact carries its own `clamps` (the intervention) and `summary` (the shown
 * explanation), and this synthesizes one scenario per entry against a single
 * shared reference world. When a workspace has no clamp-bearing saved scenarios
 * (e.g. storybook fixtures), it falls back to one generic `do()` on the primary
 * treatment so the DAG is still demonstrable. Gated by `NODE_ENV` at the
 * call site.
 */
import type { CausalEdge, Construct, Indicator, Stage6Data } from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import type { Stage6SimulationResult } from "../intervention-dag-types";
import type {
  EdgeDrift,
  IndicatorSeries,
  SelfEffect,
  Stage6VisualizationExt,
} from "./contract-extension";
import type { SimulateFn } from "./simulate-input";

const HORIZON = 60;
const DO_DAY = 7;
const DO_VALUE = 0.95;
const NS = 22; // transfer-curve samples
const clamp01 = (v: number): number => Math.max(0, Math.min(1, v));
const round4 = (v: number): number => Number(v.toFixed(4));

const SHAPE = {
  linear: (u: number) => u,
  hill: (u: number) => (u ** 3 / (0.125 + u ** 3)) * 1.125,
  mult: (u: number) => u ** 1.9,
} as const;
type Form = keyof typeof SHAPE;
const FORMS: Form[] = ["linear", "hill", "mult"];

function seed01(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 1000) / 1000;
}
function mulberry32(a: number): () => number {
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const gaussOf = (r: () => number) =>
  Math.sqrt(-2 * Math.log(Math.max(1e-9, r()))) * Math.cos(2 * Math.PI * r());
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

const DAYS = Array.from({ length: HORIZON + 1 }, (_, i) => i);

/** Element of Stage 6's `saved_scenarios` (the contract type isn't re-exported by name). */
type SavedScenario = NonNullable<Stage6Data["saved_scenarios"]>[number];

/** A timed latent clamp the mock can read off a saved scenario. */
interface MockClamp {
  variable: string;
  value: number;
  from_day?: number;
}
/** `saved_scenarios` entries in the demo data carry clamps the contract doesn't type. */
type DemoSavedScenario = SavedScenario & { clamps?: MockClamp[] };

interface ScenarioSpec {
  id: string;
  query?: string;
  /** Shown explanation; falls back to a generated blurb when absent. */
  summary?: string;
  clamps: MockClamp[];
}

/** One synthesized scenario: its result + the explanation shown beside it. */
export interface MockScenario {
  id: string;
  result: Stage6SimulationResult;
  blurb: string;
  query?: string;
}
/** All scenarios share a single synthesized reference world. */
export interface MockScenarios {
  baseline: MockScenario;
  interventions: MockScenario[];
}

export function synthesizeMockScenarios(
  constructs: Construct[],
  edges: CausalEdge[],
  indicators: Indicator[],
  outcome: string,
  savedScenarios?: SavedScenario[] | null,
): MockScenarios {
  // Full projected estimation graph — no cone restriction. Every passed-in
  // construct gets a synthesized trajectory; `present` only guards edges against
  // referencing a non-construct node.
  const present = new Set(constructs.map((c) => c.name));
  const inGraph = (name: string) => present.has(name);

  // forward adjacency over the graph's edges → propagate a clamp to its descendants
  const succ = new Map<string, string[]>();
  for (const e of edges) {
    if (!inGraph(e.cause) || !inGraph(e.effect) || e.cause === e.effect) continue;
    const list = succ.get(e.cause);
    if (list) list.push(e.effect);
    else succ.set(e.cause, [e.effect]);
  }

  const isEndo = new Map(constructs.map((c) => [c.name, c.role === "endogenous"]));
  const isVarying = new Map(
    constructs.map((c) => [c.name, c.temporal_status === "time_varying"]),
  );
  const baselineLevel: Record<string, number> = {};
  const tau: Record<string, number> = {};
  const quartic: Record<string, number> = {};
  for (const c of constructs) {
    baselineLevel[c.name] = 0.35 + 0.3 * seed01(c.name);
    tau[c.name] = 8 + 20 * seed01(`${c.name}:tau`);
    quartic[c.name] = c.name === outcome ? 0.12 : 0;
  }

  const exoPath = (name: string, t: number) =>
    baselineLevel[name] +
    (isVarying.get(name) ? 0.04 * Math.sin(t / 9 + baselineLevel[name] * 7) : 0);

  // ── shared reference world (the no-intervention baseline) ──────────────────
  const reference: Record<string, number[]> = {};
  const node_realized: Record<string, number[]> = {};
  for (const c of constructs) {
    const endo = isEndo.get(c.name) ?? false;
    reference[c.name] = DAYS.map((t) => round4(endo ? baselineLevel[c.name] : exoPath(c.name, t)));
    if (endo) {
      const rng = mulberry32(2166136261 ^ Math.floor(seed01(`realized:${c.name}`) * 1e9));
      let e = 0;
      node_realized[c.name] = reference[c.name].map((v) => {
        e = 0.6 * e + 0.4 * gaussOf(rng);
        return round4(clamp01(v + 0.05 * e));
      });
    }
  }

  /** Apply clamps to the reference world: pin each clamped node, ramp its descendants. */
  const buildAction = (clamps: MockClamp[]): Record<string, number[]> => {
    const clampVars = new Set(clamps.map((c) => c.variable));
    const moved = new Set<string>(clampVars);
    const stack = [...clampVars];
    while (stack.length) {
      const node = stack.pop() as string;
      for (const nxt of succ.get(node) ?? [])
        if (!moved.has(nxt)) {
          moved.add(nxt);
          stack.push(nxt);
        }
    }
    const action: Record<string, number[]> = {};
    for (const [node, series] of Object.entries(reference)) action[node] = [...series];
    for (const clamp of clamps) {
      const from = clamp.from_day ?? DO_DAY;
      const delta = clamp.value - (reference[clamp.variable]?.[0] ?? 0);
      for (const node of moved) {
        if (clampVars.has(node)) continue;
        const series = action[node];
        if (!series) continue;
        for (let t = 0; t < series.length; t++) {
          const ramp = t < from ? 0 : Math.min(1, (t - from) / 20);
          series[t] = round4(clamp01(series[t] + 0.25 * delta * ramp));
        }
      }
    }
    for (const clamp of clamps) {
      const from = clamp.from_day ?? DO_DAY;
      const pinned = action[clamp.variable];
      if (!pinned) continue;
      for (let t = 0; t < pinned.length; t++)
        if (t >= from) pinned[t] = round4(clamp01(clamp.value));
    }
    return action;
  };

  // ── per-edge / per-node config (level-independent), evaluated at any world ──
  const edgeConfigs: { cause: string; effect: string; form: Form; c: (u: number) => number }[] = [];
  let idx = 0;
  for (const e of edges) {
    if (e.cause === e.effect || !inGraph(e.cause) || !inGraph(e.effect)) continue;
    const form = FORMS[idx % FORMS.length];
    const weight = 0.3 + 0.5 * seed01(`${e.cause}>${e.effect}`);
    idx++;
    const base0 = SHAPE[form](clamp01(baselineLevel[e.cause] ?? 0.5));
    edgeConfigs.push({
      cause: e.cause,
      effect: e.effect,
      form,
      c: (u: number) => weight * (SHAPE[form](clamp01(u)) - base0),
    });
  }
  const edgeDriftFor = (levels: Record<string, number[]>): EdgeDrift[] =>
    edgeConfigs.map(({ cause, effect, form, c }) => ({
      cause,
      effect,
      form,
      transfer: Array.from({ length: NS + 1 }, (_, i) => round4(c(i / NS))),
      contribution: DAYS.map((t) => round4(c(levels[cause]?.[t] ?? 0))),
      driver_level: DAYS.map((t) => round4(clamp01(levels[cause]?.[t] ?? 0))),
    }));

  const selfConfigs = constructs
    .filter((c) => isEndo.get(c.name) ?? false)
    .map((c) => {
      const k = 1 / tau[c.name];
      const c0 = baselineLevel[c.name];
      const q = quartic[c.name];
      return { node: c.name, c0, sd: (s: number) => -(k * (s - c0) + q * (s - c0) ** 3) };
    });
  const selfEffectsFor = (levels: Record<string, number[]>): SelfEffect[] =>
    selfConfigs.map(({ node, c0, sd }) => ({
      node,
      transfer: Array.from({ length: NS + 1 }, (_, i) => round4(sd(i / NS))),
      contribution: DAYS.map((t) => round4(sd(levels[node]?.[t] ?? c0))),
      level: DAYS.map((t) => round4(clamp01(levels[node]?.[t] ?? c0))),
    }));

  // indicator configs: factual fit + sampled observations are shared; cf varies by world
  const indMean = (load: number, off: number, eta: number, binary: boolean) =>
    binary ? sigmoid(load * (eta - 0.5) + off) : clamp01(off + load * (eta - 0.5));
  const indicatorConfigs = indicators
    .filter((ind) => inGraph(ind.construct_name))
    .map((ind) => {
      const binary = /bin|bool|count|ordinal|categor/i.test(ind.measurement_dtype ?? "");
      const r = mulberry32(
        2166136261 ^ Math.floor(seed01(`${ind.construct_name}:${ind.name}`) * 1e9),
      );
      const load = binary
        ? 4 + 2 * seed01(`${ind.name}:load`)
        : 0.6 + 0.4 * seed01(`${ind.name}:load`);
      const off = binary ? -0.5 : 0.45 + 0.1 * (seed01(`${ind.name}:off`) - 0.5);
      const sd = 0.06;
      const every = 1 + Math.floor(3 * seed01(`${ind.name}:every`));
      const refSeries = DAYS.map((t) =>
        indMean(load, off, reference[ind.construct_name]?.[t] ?? 0.5, binary),
      );
      const observed: { t: number; v: number }[] = [];
      for (const t of DAYS) {
        if (t % every !== 0) continue;
        observed.push(
          binary
            ? { t, v: r() < refSeries[t] ? 1 : 0 }
            : { t, v: round4(clamp01(refSeries[t] + sd * gaussOf(r))) },
        );
      }
      return {
        construct: ind.construct_name,
        id: ind.name,
        type: binary ? ("binary" as const) : ("cont" as const),
        sd,
        observed,
        ref: refSeries.map(round4),
        load,
        off,
        binary,
      };
    });
  const indicatorsFor = (levels: Record<string, number[]>): IndicatorSeries[] =>
    indicatorConfigs.map(({ load, off, binary, ...base }) => ({
      ...base,
      cf: DAYS.map((t) => round4(indMean(load, off, levels[base.construct]?.[t] ?? 0.5, binary))),
    }));

  const zeros = DAYS.map(() => 0);
  const baselineLevelOutcome = baselineLevel[outcome] ?? 0;

  const vizFor = (action: Record<string, number[]>): Stage6VisualizationExt => ({
    reference_node_trajectories: reference,
    action_node_trajectories: action,
    node_effect_trajectories: Object.fromEntries(
      constructs.map((c) => [
        c.name,
        DAYS.map((t) => round4((action[c.name]?.[t] ?? 0) - (reference[c.name]?.[t] ?? 0))),
      ]),
    ),
    start_state: null,
    node_realized,
    edge_drift: edgeDriftFor(action),
    self_effects: selfEffectsFor(action),
    indicators: indicatorsFor(action),
  });

  // ── baseline scenario ──────────────────────────────────────────────────────
  const baselineViz: Stage6VisualizationExt = {
    ...vizFor(reference),
    action_node_trajectories: Object.fromEntries(
      constructs.map((c) => [c.name, reference[c.name].slice()]),
    ),
    node_effect_trajectories: Object.fromEntries(
      constructs.map((c) => [c.name, zeros.slice()]),
    ),
  };
  const baselineResult: Stage6SimulationResult = {
    start: {
      kind: "baseline",
      time_index: null,
      time: null,
      state_source: "baseline_steady_state",
    },
    clamps: [],
    outcome,
    estimand: "trajectory",
    reference_mean: baselineLevelOutcome,
    summary: { mean: 0, median: 0, lower_95: 0, upper_95: 0, prob_positive: 0.5 },
    effect_trajectory: DAYS.map((t) => ({ day: t, effect: 0 })),
    visualization: baselineViz,
    manifest_effects: null,
    warnings: [],
  };

  // ── intervention scenarios (data-driven, with a generic fallback) ──────────
  const specs = deriveSpecs(savedScenarios, constructs, outcome, present);
  const interventions: MockScenario[] = specs.map((spec) => {
    const action = buildAction(spec.clamps);
    const effectTrajectory = DAYS.map((t) => ({
      day: t,
      effect: round4((action[outcome]?.[t] ?? 0) - (reference[outcome]?.[t] ?? 0)),
    }));
    const mean = effectTrajectory[HORIZON]?.effect ?? 0;
    const result: Stage6SimulationResult = {
      start: {
        kind: "baseline",
        time_index: null,
        time: null,
        state_source: "baseline_steady_state",
      },
      clamps: spec.clamps.map((c) => ({
        variable: c.variable,
        mode: "set",
        value: c.value,
        from_day: c.from_day ?? DO_DAY,
      })),
      outcome,
      estimand: "trajectory",
      reference_mean: baselineLevelOutcome,
      summary: {
        mean: round4(mean),
        median: round4(mean),
        lower_95: round4(mean - 0.08),
        upper_95: round4(mean + 0.08),
        prob_positive: mean >= 0 ? 0.94 : 0.06,
      },
      effect_trajectory: effectTrajectory,
      visualization: vizFor(action),
      manifest_effects: null,
      warnings: [],
    };
    return { id: spec.id, result, blurb: spec.summary ?? blurbFor(result), query: spec.query };
  });

  return {
    baseline: { id: "dev-mock-baseline", result: baselineResult, blurb: blurbFor(baselineResult) },
    interventions,
  };
}

/** Build scenario specs from the data's clamp-bearing saved scenarios, else one generic do(). */
function deriveSpecs(
  savedScenarios: SavedScenario[] | null | undefined,
  constructs: Construct[],
  outcome: string,
  present: Set<string>,
): ScenarioSpec[] {
  const fromData = (savedScenarios ?? []).flatMap((scenario, i) => {
    const clamps = ((scenario as DemoSavedScenario).clamps ?? []).filter((c) =>
      present.has(c.variable),
    );
    if (clamps.length === 0) return [];
    return [
      {
        id: `saved-${i}`,
        query: scenario.query,
        summary: scenario.summary ?? undefined,
        clamps,
      },
    ];
  });
  if (fromData.length > 0) return fromData;

  // Fallback narrative (storybook / clampless fixtures): the patient's "should I stop?"
  // taper — do(adherence = 0) from day 0, propagating adherence → serotonergic_exposure
  // → outcome — read against the no-intervention "stay on it" reference.
  if (present.has("adherence") && outcome !== "adherence") {
    return [{ id: "taper", clamps: [{ variable: "adherence", value: 0, from_day: 0 }] }];
  }
  const treatment =
    constructs.find(
      (c) => c.name !== outcome && c.role === "endogenous" && c.temporal_status === "time_varying",
    )?.name ?? constructs.find((c) => c.name !== outcome)?.name;
  return treatment
    ? [{ id: "generic", clamps: [{ variable: treatment, value: DO_VALUE, from_day: DO_DAY }] }]
    : [];
}

const signed = (v: number): string => `${v >= 0 ? "+" : ""}${v.toFixed(2)}`;
const pretty = (name: string): string => name.replace(/_/g, " ");

function blurbFor(result: Stage6SimulationResult): string {
  if (result.clamps.length === 0) {
    return [
      "**No intervention — staying on the medication.**",
      `Adherence holds at its current level, so ${pretty(result.outcome)} stays at its medicated set-point all the way out to day ${HORIZON} — this is the factual "keep taking it" world, with the measured indicators scattering around the model's fit to it. Every intervention below is read as a deviation from this reference.`,
    ].join(" ");
  }
  const action = result.clamps
    .map((clamp) => `${pretty(clamp.variable)} to ${(clamp.value ?? clamp.amount ?? 0).toFixed(2)}`)
    .join(" and ");
  const fromDay = result.clamps[0]?.from_day ?? 0;
  const mean = result.summary.mean;
  return [
    `**Why this intervention.** Clamping ${action} from day ${fromDay} fixes the intervened levers and lets the effect propagate through their descendants toward ${pretty(result.outcome)}.`,
    `**What happens.** Relative to staying on the medication, ${pretty(result.outcome)} settles to ${signed(mean)} SD by day ${HORIZON} (95% CI [${signed(result.summary.lower_95)}, ${signed(result.summary.upper_95)}]); edge thickness tracks each channel's live drift contribution at the playhead.`,
  ].join(" ");
}

/** Wrap the synthesized scenarios as refinement messages so `buildStage6Scenarios` picks them up. */
export function buildDevMockMessages({ baseline, interventions }: MockScenarios): UIMessage[] {
  const mkMessages = (scenario: MockScenario): UIMessage[] => {
    const messages: UIMessage[] = [];
    if (scenario.query) {
      messages.push({
        id: `${scenario.id}-q`,
        role: "user",
        parts: [{ type: "text", text: scenario.query }],
      });
    }
    messages.push({
      id: scenario.id,
      role: "assistant",
      parts: [
        { type: "text", text: scenario.blurb },
        {
          type: "dynamic-tool",
          toolCallId: scenario.id,
          toolName: "simulate",
          state: "output-available",
          input: { query: { horizon_days: HORIZON } },
          output: scenario.result,
        },
      ],
    });
    return messages;
  };
  // Baseline first (oldest), interventions after (the newest is the default focus).
  return [baseline, ...interventions].flatMap(mkMessages);
}

/**
 * Stand-in for the backend `simulate` tool: pins each set-clamped construct from
 * the clamp onset and nudges everything downstream. Spreads the base
 * visualization so extension fields (drift, self, indicators) carry through.
 */
export function makeMockSimulate(base: Stage6SimulationResult): SimulateFn {
  return async (input) => {
    await new Promise((resolve) => setTimeout(resolve, 350));
    const viz = base.visualization;
    if (!viz) return base;
    const reference = viz.reference_node_trajectories ?? {};
    // Re-simulate from the reference world so do() is one-at-a-time (a new clamp
    // replaces the scenario rather than stacking on its existing movement).
    const action: Record<string, number[]> = {};
    for (const [node, series] of Object.entries(reference)) {
      action[node] = series ? [...series] : [];
    }
    for (const clamp of input.clamps) {
      if (clamp.mode !== "set" || clamp.value == null) continue;
      const from = clamp.from_day ?? 0;
      const baseVal = reference[clamp.variable]?.[0] ?? 0;
      const delta = clamp.value - baseVal;
      const pinned = action[clamp.variable];
      if (pinned) for (let t = 0; t < pinned.length; t++) if (t >= from) pinned[t] = clamp.value;
      for (const [node, series] of Object.entries(action)) {
        if (node === clamp.variable) continue;
        for (let t = 0; t < series.length; t++) {
          const ramp = t < from ? 0 : Math.min(1, (t - from) / 20);
          series[t] = Number((series[t] + 0.25 * delta * ramp).toFixed(4));
        }
      }
    }
    return {
      ...base,
      clamps: input.clamps,
      visualization: { ...viz, action_node_trajectories: action },
    };
  };
}
