import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  MODEL_SPEC_ADMISSION_EVENT_PREFIX,
  type ModelSpecAdmissionCheckResult,
  type ModelSpecAdmissionCoupledRecheck,
  type ModelSpecAdmissionEventRecord,
  type ModelSpecAdmissionParameter,
  type ModelSpecAdmissionPlan,
  type ModelSpecAdmissionReplayState,
  EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE,
  applyModelSpecAdmissionEvent,
  parseModelSpecAdmissionEvent,
} from "@/lib/model-spec-admission-runtime";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import { useEffect, useState } from "react";
import { outputStoryDecorators } from "../output-story-helpers";
import { OutputStoryTemplate } from "../output-story-template";
import { ModelSpecAdmissionRunningView } from "./statistical-model-spec-running-view";

const output = TRANSITIONS.find((s) => s.id === "statistical_model_spec")!;

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/Admission",
  component: ModelSpecAdmissionRunningView,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof ModelSpecAdmissionRunningView>;

export default meta;

type Story = StoryObj<typeof meta>;

// Illustrative authored priors keyed on the semantic-binding parameter names
// (rho_ persistence, sigma_ process SD, beta_ edge weight, lambda_ loading,
// obs_sd_ residual SD, t0_ initial state, self_limit_ / obs_shape_ dynamics).
function priorFor(name: string): ModelSpecAdmissionParameter {
  if (name.startsWith("t0_mean_"))
    return { name, distribution: "Normal", params: { mu: 0, sigma: 1 } };
  if (name.startsWith("t0_sd_")) return { name, distribution: "HalfNormal", params: { sigma: 1 } };
  if (name.startsWith("rho_")) return { name, distribution: "Beta", params: { alpha: 2, beta: 2 } };
  if (name.startsWith("sigma_")) return { name, distribution: "HalfNormal", params: { sigma: 1 } };
  if (name.startsWith("self_limit_"))
    return { name, distribution: "HalfNormal", params: { sigma: 0.5 } };
  if (name.startsWith("lambda_"))
    return { name, distribution: "Normal", params: { mu: 1, sigma: 0.5 } };
  if (name.startsWith("obs_sd_")) return { name, distribution: "HalfNormal", params: { sigma: 1 } };
  if (name.startsWith("obs_shape_"))
    return { name, distribution: "Gamma", params: { alpha: 2, beta: 2 } };
  return { name, distribution: "Normal", params: { mu: 0, sigma: 1 } };
}

const PLAN: ModelSpecAdmissionPlan = {
  max_attempts: 4,
  constructs: [
    {
      name: "cyp2c19_metabolizer_status",
      label: "CYP2C19 metabolizer status",
      parents: [],
      indicators: ["genotype_phenotype"],
      parameters: ["t0_mean_cyp2c19_metabolizer_status", "t0_sd_cyp2c19_metabolizer_status"].map(
        priorFor,
      ),
    },
    {
      name: "recurrence_vulnerability",
      parents: [],
      indicators: ["prior_episode_count", "years_since_first_episode"],
      parameters: [
        "rho_recurrence_vulnerability",
        "sigma_recurrence_vulnerability",
        "self_limit_recurrence_vulnerability",
        "lambda_years_since_first_episode_recurrence_vulnerability",
      ].map(priorFor),
    },
    {
      name: "episode_phase",
      parents: ["recurrence_vulnerability"],
      indicators: ["days_since_remission", "current_phase_label"],
      parameters: [
        "rho_episode_phase",
        "sigma_episode_phase",
        "beta_recurrence_vulnerability_episode_phase",
      ].map(priorFor),
    },
    {
      name: "stress_load",
      parents: [],
      indicators: ["acute_stressor_events", "journal_stress_rating"],
      parameters: ["rho_stress_load", "sigma_stress_load", "obs_sd_journal_stress_rating"].map(
        priorFor,
      ),
    },
    {
      name: "sleep_disturbance",
      parents: ["stress_load", "physical_health", "symptom_burden"],
      indicators: ["sleep_onset_latency_min", "wake_after_sleep_onset_min"],
      parameters: [
        "rho_sleep_disturbance",
        "sigma_sleep_disturbance",
        "beta_stress_load_sleep_disturbance",
        "beta_physical_health_sleep_disturbance",
        "beta_symptom_burden_sleep_disturbance",
        "obs_shape_sleep_onset_latency_min",
      ].map(priorFor),
    },
    {
      name: "symptom_burden",
      parents: ["sleep_disturbance", "behavioral_activation", "social_support"],
      indicators: ["state_of_mind_valence", "journal_negative_affect"],
      parameters: [
        "rho_symptom_burden",
        "sigma_symptom_burden",
        "beta_sleep_disturbance_symptom_burden",
        "beta_behavioral_activation_symptom_burden",
        "beta_social_support_symptom_burden",
      ].map(priorFor),
      closing_edges: ["symptom_burden->sleep_disturbance"],
    },
    {
      name: "dose_schedule",
      parents: ["episode_phase", "stress_load", "symptom_burden"],
      indicators: ["prescribed_dose_mg", "dose_change_event"],
      parameters: [
        "rho_dose_schedule",
        "sigma_dose_schedule",
        "beta_episode_phase_dose_schedule",
        "beta_stress_load_dose_schedule",
        "beta_symptom_burden_dose_schedule",
      ].map(priorFor),
    },
    {
      name: "escitalopram_exposure",
      parents: ["dose_schedule", "access_supply", "cyp2c19_metabolizer_status"],
      indicators: ["proportion_days_covered", "pharmacy_fill_count"],
      parameters: [
        "rho_escitalopram_exposure",
        "sigma_escitalopram_exposure",
        "beta_dose_schedule_escitalopram_exposure",
        "beta_cyp2c19_metabolizer_status_escitalopram_exposure",
      ].map(priorFor),
    },
  ],
  edges: [
    { cause: "recurrence_vulnerability", effect: "episode_phase" },
    { cause: "stress_load", effect: "sleep_disturbance" },
    { cause: "symptom_burden", effect: "sleep_disturbance" },
    { cause: "sleep_disturbance", effect: "symptom_burden" },
    { cause: "episode_phase", effect: "dose_schedule" },
    { cause: "stress_load", effect: "dose_schedule" },
    { cause: "symptom_burden", effect: "dose_schedule" },
    { cause: "dose_schedule", effect: "escitalopram_exposure" },
    { cause: "cyp2c19_metabolizer_status", effect: "escitalopram_exposure" },
  ],
};

function admissionEvent(
  type: string,
  payload: Record<string, unknown> = {},
): ModelSpecAdmissionEventRecord {
  return {
    event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}${type}`,
    occurred: new Date().toISOString(),
    payload: { context_id: "statistical-model-spec", type, ...payload },
  };
}

// Illustrative per-check wall-clock. The reachability battery runs particle /
// Diffrax prior-predictive simulations, so a few hundred ms to a couple seconds
// is realistic. Derived deterministically from the check identity so the badge
// stays stable across the animated replay's re-renders.
function checkDuration(check: string, target: string): number {
  const seed = [...`${check}|${target}`].reduce(
    (acc, char) => (acc * 31 + char.charCodeAt(0)) % 100003,
    7,
  );
  return 180 + (seed % 2600);
}

function passed(
  check: string,
  target: string,
  value: string,
  band: string,
): ModelSpecAdmissionCheckResult {
  return {
    check,
    target,
    value,
    band,
    duration_ms: checkDuration(check, target),
    passed: true,
    note: "",
    mode: "soft",
  };
}

function failed(
  check: string,
  target: string,
  value: string,
  band: string,
  note: string,
  mode: "hard" | "soft",
  diagnosis: string[] = [],
): ModelSpecAdmissionCheckResult {
  return {
    check,
    target,
    value,
    band,
    duration_ms: checkDuration(check, target),
    passed: false,
    note,
    mode,
    diagnosis,
  };
}

function report(
  name: string,
  attempt: number,
  outcome: string,
  admitted: boolean,
  results: ModelSpecAdmissionCheckResult[],
  annotations: string[] = [],
  coupledRecheck?: ModelSpecAdmissionCoupledRecheck,
): ModelSpecAdmissionEventRecord {
  return admissionEvent("construct_report", {
    name,
    attempt,
    outcome,
    admitted,
    annotations,
    results,
    coupled_recheck: coupledRecheck,
  });
}

const CYP_REPORT = report("cyp2c19_metabolizer_status", 1, "ADMITTED", true, [
  passed("C1a finiteness", "cyp2c19_metabolizer_status", "nonfinite 0.0%", "0%"),
  passed(
    "C1b confinement",
    "cyp2c19_metabolizer_status",
    "P(late/early amplitude > 5) 0.0%",
    "<1%",
  ),
  passed("C2 latent scale", "cyp2c19_metabolizer_status", "median sd 0.87", "[0.33, 3.00]"),
  passed("C5a location reach", "genotype_phenotype", "obs quantiles in pp band: yes", "all inside"),
]);

const RECURRENCE_REPORT = report(
  "recurrence_vulnerability",
  1,
  "ADMITTED with accepted consequences",
  true,
  [
    passed("C1a finiteness", "recurrence_vulnerability", "nonfinite 0.0%", "0%"),
    passed(
      "C1b confinement",
      "recurrence_vulnerability",
      "P(late/early amplitude > 5) 0.2%",
      "<1%",
    ),
    failed(
      "C3 resolvability",
      "recurrence_vulnerability",
      "prior tau median 740.00 d; 12% in window",
      "cadence/3 <= tau <= span/4 = [0.33, 182.50] d",
      "the timescale is slower than this sampling window can resolve.",
      "soft",
      [
        "accepted because this construct represents stable chronic vulnerability, not a fast state.",
      ],
    ),
    passed("C5b width", "prior_episode_count", "IQR ratio prior-pred/data 1.70", "[0.33, 50]"),
  ],
  [
    "recurrence_vulnerability: its timescale sits outside what this sampling design resolves; trajectory statements are prior-set.",
  ],
);

const EPISODE_REPORT = report("episode_phase", 1, "ADMITTED", true, [
  passed("C1a finiteness", "episode_phase", "nonfinite 0.0%", "0%"),
  passed("C2 latent scale", "episode_phase", "median sd 1.16", "[0.42, 3.76]"),
  passed(
    "C4b edge overwhelm",
    "episode_phase",
    "edge path displacement / child scale: median 31.0%",
    "median <= 95%",
  ),
  passed("C5c transmission", "current_phase_label", "signal IQR / data IQR 82%", ">= 20%"),
]);

const STRESS_FIRST_REPORT = report(
  "stress_load",
  1,
  "BLOCKED - hard failure: revise the fragment (no override)",
  false,
  [
    passed("C1a finiteness", "stress_load", "nonfinite 0.0%", "0%"),
    failed(
      "C5a location reach",
      "journal_stress_rating",
      "obs quantiles in pp [1,99]% band [0.1, 1.8]: NO",
      "all inside",
      "the prior predictive cannot reach the location where journal_stress_rating actually lives.",
      "hard",
      [
        "observed median sits 4.4 units above the predictive center.",
        "raise the observation intercept or widen the manifest mean prior.",
      ],
    ),
    failed(
      "C5c transmission",
      "journal_stress_rating",
      "signal IQR / data IQR 7%",
      ">= 20%",
      "the link transmits little of the latent variation.",
      "soft",
      ["loading prior is too close to zero for the observed variation."],
    ),
  ],
);

const STRESS_SECOND_REPORT = report("stress_load", 2, "ADMITTED", true, [
  passed("C1a finiteness", "stress_load", "nonfinite 0.0%", "0%"),
  passed("C1b confinement", "stress_load", "P(late/early amplitude > 5) 0.6%", "<1%"),
  passed("C2 latent scale", "stress_load", "median sd 1.92", "[0.71, 6.39]"),
  passed(
    "C5a location reach",
    "journal_stress_rating",
    "obs quantiles in pp [1,99]% band: yes",
    "all inside",
  ),
  passed("C5c transmission", "journal_stress_rating", "signal IQR / data IQR 44%", ">= 20%"),
]);

const SLEEP_PARTIAL_REPORT = report(
  "sleep_disturbance",
  1,
  "NEEDS DECISION - revise the fragment or accept the consequence (C4c saturation)",
  false,
  [
    passed("C1a finiteness", "sleep_disturbance", "nonfinite 0.0%", "0%"),
    passed("C2 latent scale", "sleep_disturbance", "median sd 2.20", "[0.90, 8.10]"),
    passed(
      "C4b edge overwhelm",
      "sleep_disturbance",
      "edge path displacement / child scale: median 42.0%",
      "median <= 95%",
    ),
    failed(
      "C4c saturation",
      "stress_load->sleep_disturbance",
      "EC50 median 9.10 vs parent 10-90% [-1.30, 2.80]",
      "EC50 inside parent range",
      "the saturating edge is not exercised over the parent's prior range.",
      "soft",
      ["drop the Hill form or shift EC50 into the realized parent range."],
    ),
    passed(
      "C5a location reach",
      "sleep_onset_latency_min",
      "obs quantiles in pp band: yes",
      "all inside",
    ),
  ],
);

const SLEEP_SECOND_REPORT = report("sleep_disturbance", 2, "ADMITTED", true, [
  passed("C1a finiteness", "sleep_disturbance", "nonfinite 0.0%", "0%"),
  passed(
    "C4c saturation",
    "stress_load->sleep_disturbance",
    "EC50 median 1.40 vs parent range [-1.30, 2.80]",
    "EC50 inside parent range",
  ),
  passed(
    "C5a location reach",
    "sleep_onset_latency_min",
    "obs quantiles in pp band: yes",
    "all inside",
  ),
]);

const SYMPTOM_RECHECK_REPORT = report(
  "symptom_burden",
  1,
  "ADMITTED",
  true,
  [
    passed("C1a finiteness", "symptom_burden", "nonfinite 0.0%", "0%"),
    passed("C2 latent scale", "symptom_burden", "median sd 1.38", "[0.45, 4.20]"),
    passed(
      "C4b edge overwhelm",
      "sleep_disturbance->symptom_burden",
      "edge path displacement / child scale: median 36.0%",
      "median <= 95%",
    ),
    passed(
      "C5a location reach",
      "state_of_mind_valence",
      "obs quantiles in pp band: yes",
      "all inside",
    ),
  ],
  [],
  {
    constructs: ["sleep_disturbance", "symptom_burden"],
    closing_edges: ["symptom_burden->sleep_disturbance"],
    results: [
      passed("C1a finiteness", "sleep_disturbance", "nonfinite 0.0%", "0%"),
      passed("C2 latent scale", "sleep_disturbance", "median sd 2.04", "[0.90, 8.10]"),
      passed(
        "C4b edge overwhelm",
        "symptom_burden->sleep_disturbance",
        "closing-edge displacement / child scale: median 41.0%",
        "median <= 95%",
      ),
      passed(
        "C5a location reach",
        "sleep_onset_latency_min",
        "obs quantiles in pp band after feedback closure: yes",
        "all inside",
      ),
      passed("C1a finiteness", "symptom_burden", "nonfinite 0.0%", "0%"),
      passed(
        "C5a location reach",
        "state_of_mind_valence",
        "obs quantiles in pp band: yes",
        "all inside",
      ),
    ],
  },
);

const BASE_EVENTS: ModelSpecAdmissionEventRecord[] = [
  admissionEvent("plan", PLAN as unknown as Record<string, unknown>),
  admissionEvent("construct_started", { construct: "cyp2c19_metabolizer_status", attempt: 1 }),
  admissionEvent("construct_checking", { construct: "cyp2c19_metabolizer_status", attempt: 1 }),
  CYP_REPORT,
  admissionEvent("construct_started", { construct: "recurrence_vulnerability", attempt: 1 }),
  admissionEvent("construct_checking", { construct: "recurrence_vulnerability", attempt: 1 }),
  RECURRENCE_REPORT,
  admissionEvent("construct_started", { construct: "episode_phase", attempt: 1 }),
  admissionEvent("construct_checking", { construct: "episode_phase", attempt: 1 }),
  EPISODE_REPORT,
  admissionEvent("construct_started", { construct: "stress_load", attempt: 1 }),
  admissionEvent("construct_checking", { construct: "stress_load", attempt: 1 }),
  STRESS_FIRST_REPORT,
  admissionEvent("construct_started", { construct: "stress_load", attempt: 2 }),
  admissionEvent("construct_checking", { construct: "stress_load", attempt: 2 }),
  STRESS_SECOND_REPORT,
  admissionEvent("construct_started", { construct: "sleep_disturbance", attempt: 1 }),
  admissionEvent("construct_checking", { construct: "sleep_disturbance", attempt: 1 }),
];

function replay(events: readonly ModelSpecAdmissionEventRecord[]): ModelSpecAdmissionReplayState {
  return events.reduce<ModelSpecAdmissionReplayState>((state, raw) => {
    const event = parseModelSpecAdmissionEvent(raw);
    return event ? applyModelSpecAdmissionEvent(state, event) : state;
  }, EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE);
}

function AnimatedAdmission() {
  const [state, setState] = useState<ModelSpecAdmissionReplayState>(() =>
    replay(BASE_EVENTS.slice(0, 1)),
  );

  useEffect(() => {
    let current = EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE;
    let index = 0;

    const tick = () => {
      const raw = BASE_EVENTS[index];
      const parsed = parseModelSpecAdmissionEvent(raw);
      if (parsed) {
        current = applyModelSpecAdmissionEvent(current, parsed);
        setState(current);
      }
      index = (index + 1) % BASE_EVENTS.length;
      if (index === 0) {
        current = EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE;
      }
    };

    tick();
    const timer = window.setInterval(tick, 900);
    return () => window.clearInterval(timer);
  }, []);

  return <ModelSpecAdmissionRunningView state={state} />;
}

export const AdmissionReplay: Story = {
  args: { state: null },
  render: () => (
    <OutputStoryTemplate output={output} status="running" runningContent={<AnimatedAdmission />} />
  ),
};

export const MidRun: Story = {
  args: { state: null },
  render: () => (
    <OutputStoryTemplate
      output={output}
      status="running"
      runningContent={<ModelSpecAdmissionRunningView state={replay(BASE_EVENTS)} />}
    />
  ),
};

export const NeedsRevision: Story = {
  args: { state: null },
  render: () => (
    <OutputStoryTemplate
      output={output}
      status="running"
      runningContent={<ModelSpecAdmissionRunningView state={replay(BASE_EVENTS.slice(0, 13))} />}
    />
  ),
};

export const CoupledSubsystemRecheck: Story = {
  args: { state: null },
  render: () => {
    const events = [
      ...BASE_EVENTS,
      SLEEP_PARTIAL_REPORT,
      admissionEvent("construct_started", { construct: "sleep_disturbance", attempt: 2 }),
      admissionEvent("construct_checking", { construct: "sleep_disturbance", attempt: 2 }),
      SLEEP_SECOND_REPORT,
      admissionEvent("construct_started", { construct: "symptom_burden", attempt: 1 }),
      admissionEvent("construct_checking", { construct: "symptom_burden", attempt: 1 }),
      SYMPTOM_RECHECK_REPORT,
    ];
    return (
      <OutputStoryTemplate
        output={output}
        status="running"
        runningContent={<ModelSpecAdmissionRunningView state={replay(events)} />}
      />
    );
  },
};

export const CompletedAdmission: Story = {
  args: { state: null },
  render: () => {
    const allEvents = [
      ...BASE_EVENTS,
      SLEEP_PARTIAL_REPORT,
      admissionEvent("construct_started", { construct: "sleep_disturbance", attempt: 2 }),
      admissionEvent("construct_checking", { construct: "sleep_disturbance", attempt: 2 }),
      SLEEP_SECOND_REPORT,
      admissionEvent("construct_started", { construct: "symptom_burden", attempt: 1 }),
      admissionEvent("construct_checking", { construct: "symptom_burden", attempt: 1 }),
      SYMPTOM_RECHECK_REPORT,
      admissionEvent("construct_started", { construct: "dose_schedule", attempt: 1 }),
      admissionEvent("construct_checking", { construct: "dose_schedule", attempt: 1 }),
      report("dose_schedule", 1, "ADMITTED", true, [
        passed("C1a finiteness", "dose_schedule", "nonfinite 0.0%", "0%"),
        passed(
          "C4b edge overwhelm",
          "symptom_burden->dose_schedule",
          "edge path displacement / child scale: median 22.0%",
          "median <= 95%",
        ),
        passed(
          "C5a location reach",
          "prescribed_dose_mg",
          "obs quantiles in pp band: yes",
          "all inside",
        ),
      ]),
      admissionEvent("construct_started", { construct: "escitalopram_exposure", attempt: 1 }),
      admissionEvent("construct_checking", { construct: "escitalopram_exposure", attempt: 1 }),
      report("escitalopram_exposure", 1, "ADMITTED", true, [
        passed("C1a finiteness", "escitalopram_exposure", "nonfinite 0.0%", "0%"),
        passed(
          "C4b edge overwhelm",
          "dose_schedule->escitalopram_exposure",
          "edge path displacement / child scale: median 29.0%",
          "median <= 95%",
        ),
        passed(
          "C5a location reach",
          "proportion_days_covered",
          "obs quantiles in pp band: yes",
          "all inside",
        ),
        passed("C5c transmission", "pharmacy_fill_count", "signal IQR / data IQR 63%", ">= 20%"),
      ]),
      admissionEvent("done"),
    ];
    return (
      <OutputStoryTemplate
        output={output}
        status="running"
        runningContent={<ModelSpecAdmissionRunningView state={replay(allEvents)} />}
      />
    );
  },
};
