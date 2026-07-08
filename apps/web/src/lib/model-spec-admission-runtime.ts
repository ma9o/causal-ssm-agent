export const MODEL_SPEC_ADMISSION_EVENT_PREFIX = "nof1-causal-lab.model-spec.admission.";

export type ModelSpecAdmissionConstructStatus =
  | "pending"
  | "active"
  | "checking"
  | "revising"
  | "admitted"
  | "admitted_with_consequences"
  | "blocked";

export interface ModelSpecAdmissionParameter {
  name: string;
  /** Prior distribution family, e.g. "Normal", "HalfNormal", "Beta". */
  distribution: string;
  /** Distribution parameters keyed by name, e.g. { mu: 0, sigma: 1 }. */
  params: Record<string, number>;
}

export interface ModelSpecAdmissionPlanConstruct {
  name: string;
  label?: string;
  parents?: string[];
  indicators?: string[];
  parameters?: ModelSpecAdmissionParameter[];
  closing_edges?: string[];
}

export interface ModelSpecAdmissionPlanEdge {
  cause: string;
  effect: string;
}

export interface ModelSpecAdmissionPlan {
  constructs: ModelSpecAdmissionPlanConstruct[];
  edges: ModelSpecAdmissionPlanEdge[];
  max_attempts: number;
}

export interface ModelSpecAdmissionCheckResult {
  check: string;
  target: string;
  value: string;
  band: string;
  /** Wall-clock time the check took to run, in milliseconds. */
  duration_ms: number;
  passed: boolean;
  note: string;
  diagnosis?: string[];
  mode?: "hard" | "soft";
}

export interface ModelSpecAdmissionCoupledRecheck {
  constructs: string[];
  closing_edges?: string[];
  results: ModelSpecAdmissionCheckResult[];
}

export interface ModelSpecAdmissionReport {
  name: string;
  attempt: number;
  outcome: string;
  admitted: boolean;
  annotations: string[];
  results: ModelSpecAdmissionCheckResult[];
  /** Priors authored for this attempt; populates the construct's "Authored parameters" table. */
  parameters?: ModelSpecAdmissionParameter[];
  coupled_recheck?: ModelSpecAdmissionCoupledRecheck;
}

export interface ModelSpecAdmissionConstructState extends ModelSpecAdmissionPlanConstruct {
  status: ModelSpecAdmissionConstructStatus;
  attempt: number;
  reports: ModelSpecAdmissionReport[];
}

export interface ModelSpecAdmissionReplayState {
  plan: ModelSpecAdmissionPlan | null;
  constructs: ModelSpecAdmissionConstructState[];
  activeConstruct: string | null;
  activeAttempt: number | null;
  phase: "planning" | "authoring" | "checking" | "done" | "failed";
  done: boolean;
  latestReport: ModelSpecAdmissionReport | null;
  error: string | null;
}

export interface ModelSpecAdmissionEventRecord {
  event?: string | null;
  occurred?: string | null;
  payload?: Record<string, unknown>;
}

export type ModelSpecAdmissionEvent =
  | { type: "plan"; plan: ModelSpecAdmissionPlan }
  | { type: "construct_started"; construct: string; attempt: number }
  | { type: "construct_checking"; construct: string; attempt: number }
  | { type: "construct_report"; report: ModelSpecAdmissionReport }
  | { type: "failed"; construct?: string; message: string }
  | { type: "done" };

export const EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE: ModelSpecAdmissionReplayState = {
  plan: null,
  constructs: [],
  activeConstruct: null,
  activeAttempt: null,
  phase: "planning",
  done: false,
  latestReport: null,
  error: null,
};

export function getModelSpecAdmissionStateQueryKey(workspaceId: string) {
  return ["pipeline", workspaceId, "model-spec-admission-state"] as const;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function numberRecord(value: unknown): Record<string, number> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, number] => typeof entry[1] === "number",
    ),
  );
}

function parseParameter(value: unknown): ModelSpecAdmissionParameter | null {
  if (!isRecord(value) || typeof value.name !== "string") return null;
  return {
    name: value.name,
    distribution: typeof value.distribution === "string" ? value.distribution : "",
    params: numberRecord(value.params),
  };
}

function parseParameters(value: unknown): ModelSpecAdmissionParameter[] {
  return Array.isArray(value)
    ? value
        .map(parseParameter)
        .filter((param): param is ModelSpecAdmissionParameter => param !== null)
    : [];
}

function parsePlanConstruct(value: unknown): ModelSpecAdmissionPlanConstruct | null {
  if (!isRecord(value) || typeof value.name !== "string") return null;
  return {
    name: value.name,
    label: typeof value.label === "string" ? value.label : undefined,
    parents: stringArray(value.parents),
    indicators: stringArray(value.indicators),
    parameters: parseParameters(value.parameters),
    closing_edges: stringArray(value.closing_edges),
  };
}

function parsePlanEdge(value: unknown): ModelSpecAdmissionPlanEdge | null {
  if (!isRecord(value) || typeof value.cause !== "string" || typeof value.effect !== "string") {
    return null;
  }
  return { cause: value.cause, effect: value.effect };
}

function parsePlan(payload: Record<string, unknown>): ModelSpecAdmissionPlan | null {
  if (!Array.isArray(payload.constructs)) return null;
  return {
    constructs: payload.constructs
      .map(parsePlanConstruct)
      .filter((construct): construct is ModelSpecAdmissionPlanConstruct => construct !== null),
    edges: Array.isArray(payload.edges)
      ? payload.edges
          .map(parsePlanEdge)
          .filter((edge): edge is ModelSpecAdmissionPlanEdge => !!edge)
      : [],
    max_attempts: typeof payload.max_attempts === "number" ? payload.max_attempts : 4,
  };
}

function parseCheckResult(value: unknown): ModelSpecAdmissionCheckResult | null {
  if (!isRecord(value) || typeof value.check !== "string") return null;
  return {
    check: value.check,
    target: typeof value.target === "string" ? value.target : "",
    value: typeof value.value === "string" ? value.value : "",
    band: typeof value.band === "string" ? value.band : "",
    duration_ms: typeof value.duration_ms === "number" ? value.duration_ms : 0,
    passed: value.passed === true,
    note: typeof value.note === "string" ? value.note : "",
    diagnosis: stringArray(value.diagnosis),
    mode: value.mode === "hard" || value.mode === "soft" ? value.mode : undefined,
  };
}

function parseReport(payload: Record<string, unknown>): ModelSpecAdmissionReport | null {
  if (typeof payload.name !== "string" || typeof payload.outcome !== "string") return null;
  const coupledRecheck = isRecord(payload.coupled_recheck)
    ? {
        constructs: stringArray(payload.coupled_recheck.constructs),
        closing_edges: stringArray(payload.coupled_recheck.closing_edges),
        results: Array.isArray(payload.coupled_recheck.results)
          ? payload.coupled_recheck.results
              .map(parseCheckResult)
              .filter((result): result is ModelSpecAdmissionCheckResult => result !== null)
          : [],
      }
    : undefined;
  return {
    name: payload.name,
    attempt: typeof payload.attempt === "number" ? payload.attempt : 1,
    outcome: payload.outcome,
    admitted: payload.admitted === true,
    annotations: stringArray(payload.annotations),
    results: Array.isArray(payload.results)
      ? payload.results
          .map(parseCheckResult)
          .filter((result): result is ModelSpecAdmissionCheckResult => result !== null)
      : [],
    parameters: Array.isArray(payload.parameters) ? parseParameters(payload.parameters) : undefined,
    coupled_recheck:
      coupledRecheck && coupledRecheck.results.length > 0 ? coupledRecheck : undefined,
  };
}

export function parseModelSpecAdmissionEvent(
  record: ModelSpecAdmissionEventRecord | null | undefined,
): ModelSpecAdmissionEvent | null {
  if (!record?.event?.startsWith(MODEL_SPEC_ADMISSION_EVENT_PREFIX) || !record.payload) return null;
  const eventType = record.event.slice(MODEL_SPEC_ADMISSION_EVENT_PREFIX.length);
  const payload = record.payload;

  if (eventType === "plan") {
    const plan = parsePlan(payload);
    return plan ? { type: "plan", plan } : null;
  }

  if (
    (eventType === "construct_started" || eventType === "construct_checking") &&
    typeof payload.construct === "string"
  ) {
    return {
      type: eventType,
      construct: payload.construct,
      attempt: typeof payload.attempt === "number" ? payload.attempt : 1,
    };
  }

  if (eventType === "construct_report") {
    const report = parseReport(payload);
    return report ? { type: "construct_report", report } : null;
  }

  if (eventType === "failed") {
    return {
      type: "failed",
      construct: typeof payload.construct === "string" ? payload.construct : undefined,
      message:
        typeof payload.message === "string" ? payload.message : "Model spec admission failed",
    };
  }

  if (eventType === "done") return { type: "done" };
  return null;
}

function statusFromReport(report: ModelSpecAdmissionReport): ModelSpecAdmissionConstructStatus {
  if (!report.admitted) return "revising";
  return report.annotations.length > 0 ? "admitted_with_consequences" : "admitted";
}

function updateConstruct(
  constructs: ModelSpecAdmissionConstructState[],
  name: string,
  update: (construct: ModelSpecAdmissionConstructState) => ModelSpecAdmissionConstructState,
): ModelSpecAdmissionConstructState[] {
  return constructs.map((construct) => (construct.name === name ? update(construct) : construct));
}

export function applyModelSpecAdmissionEvent(
  state: ModelSpecAdmissionReplayState | undefined,
  event: ModelSpecAdmissionEvent,
): ModelSpecAdmissionReplayState {
  const current = state ?? EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE;

  if (event.type === "plan") {
    return {
      ...current,
      plan: event.plan,
      phase: "planning",
      constructs: event.plan.constructs.map((construct) => ({
        ...construct,
        status: "pending",
        attempt: 0,
        reports: [],
      })),
    };
  }

  if (event.type === "construct_started" || event.type === "construct_checking") {
    const status = event.type === "construct_checking" ? "checking" : "active";
    return {
      ...current,
      activeConstruct: event.construct,
      activeAttempt: event.attempt,
      phase: status === "checking" ? "checking" : "authoring",
      constructs: updateConstruct(current.constructs, event.construct, (construct) => ({
        ...construct,
        status,
        attempt: event.attempt,
      })),
    };
  }

  if (event.type === "construct_report") {
    const report = event.report;
    return {
      ...current,
      latestReport: report,
      activeConstruct: report.admitted ? null : report.name,
      activeAttempt: report.admitted ? null : report.attempt,
      phase: report.admitted ? "authoring" : "checking",
      constructs: updateConstruct(current.constructs, report.name, (construct) => ({
        ...construct,
        status: statusFromReport(report),
        attempt: report.attempt,
        reports: [...construct.reports, report],
        parameters: report.parameters ?? construct.parameters,
      })),
    };
  }

  if (event.type === "failed") {
    return {
      ...current,
      activeConstruct: event.construct ?? current.activeConstruct,
      phase: "failed",
      error: event.message,
      constructs: event.construct
        ? updateConstruct(current.constructs, event.construct, (construct) => ({
            ...construct,
            status: "blocked",
          }))
        : current.constructs,
    };
  }

  return {
    ...current,
    activeConstruct: null,
    activeAttempt: null,
    phase: "done",
    done: true,
  };
}
