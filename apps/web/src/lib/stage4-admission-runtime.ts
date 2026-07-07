export const STAGE4_ADMISSION_EVENT_PREFIX = "nof1-causal-lab.stage4.admission.";

export type Stage4AdmissionConstructStatus =
  | "pending"
  | "active"
  | "checking"
  | "revising"
  | "admitted"
  | "admitted_with_consequences"
  | "blocked";

export interface Stage4AdmissionParameter {
  name: string;
  /** Prior distribution family, e.g. "Normal", "HalfNormal", "Beta". */
  distribution: string;
  /** Distribution parameters keyed by name, e.g. { mu: 0, sigma: 1 }. */
  params: Record<string, number>;
}

export interface Stage4AdmissionPlanConstruct {
  name: string;
  label?: string;
  parents?: string[];
  indicators?: string[];
  parameters?: Stage4AdmissionParameter[];
  closing_edges?: string[];
}

export interface Stage4AdmissionPlanEdge {
  cause: string;
  effect: string;
}

export interface Stage4AdmissionPlan {
  constructs: Stage4AdmissionPlanConstruct[];
  edges: Stage4AdmissionPlanEdge[];
  max_attempts: number;
}

export interface Stage4AdmissionCheckResult {
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

export interface Stage4AdmissionCoupledRecheck {
  constructs: string[];
  closing_edges?: string[];
  results: Stage4AdmissionCheckResult[];
}

export interface Stage4AdmissionReport {
  name: string;
  attempt: number;
  outcome: string;
  admitted: boolean;
  annotations: string[];
  results: Stage4AdmissionCheckResult[];
  /** Priors authored for this attempt; populates the construct's "Authored parameters" table. */
  parameters?: Stage4AdmissionParameter[];
  coupled_recheck?: Stage4AdmissionCoupledRecheck;
}

export interface Stage4AdmissionConstructState extends Stage4AdmissionPlanConstruct {
  status: Stage4AdmissionConstructStatus;
  attempt: number;
  reports: Stage4AdmissionReport[];
}

export interface Stage4AdmissionReplayState {
  plan: Stage4AdmissionPlan | null;
  constructs: Stage4AdmissionConstructState[];
  activeConstruct: string | null;
  activeAttempt: number | null;
  phase: "planning" | "authoring" | "checking" | "done" | "failed";
  done: boolean;
  latestReport: Stage4AdmissionReport | null;
  error: string | null;
}

export interface Stage4AdmissionEventRecord {
  event?: string | null;
  occurred?: string | null;
  payload?: Record<string, unknown>;
}

export type Stage4AdmissionEvent =
  | { type: "plan"; plan: Stage4AdmissionPlan }
  | { type: "construct_started"; construct: string; attempt: number }
  | { type: "construct_checking"; construct: string; attempt: number }
  | { type: "construct_report"; report: Stage4AdmissionReport }
  | { type: "failed"; construct?: string; message: string }
  | { type: "done" };

export const EMPTY_STAGE4_ADMISSION_REPLAY_STATE: Stage4AdmissionReplayState = {
  plan: null,
  constructs: [],
  activeConstruct: null,
  activeAttempt: null,
  phase: "planning",
  done: false,
  latestReport: null,
  error: null,
};

export function getStage4AdmissionStateQueryKey(workspaceId: string) {
  return ["pipeline", workspaceId, "stage4-admission-state"] as const;
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

function parseParameter(value: unknown): Stage4AdmissionParameter | null {
  if (!isRecord(value) || typeof value.name !== "string") return null;
  return {
    name: value.name,
    distribution: typeof value.distribution === "string" ? value.distribution : "",
    params: numberRecord(value.params),
  };
}

function parseParameters(value: unknown): Stage4AdmissionParameter[] {
  return Array.isArray(value)
    ? value.map(parseParameter).filter((param): param is Stage4AdmissionParameter => param !== null)
    : [];
}

function parsePlanConstruct(value: unknown): Stage4AdmissionPlanConstruct | null {
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

function parsePlanEdge(value: unknown): Stage4AdmissionPlanEdge | null {
  if (!isRecord(value) || typeof value.cause !== "string" || typeof value.effect !== "string") {
    return null;
  }
  return { cause: value.cause, effect: value.effect };
}

function parsePlan(payload: Record<string, unknown>): Stage4AdmissionPlan | null {
  if (!Array.isArray(payload.constructs)) return null;
  return {
    constructs: payload.constructs
      .map(parsePlanConstruct)
      .filter((construct): construct is Stage4AdmissionPlanConstruct => construct !== null),
    edges: Array.isArray(payload.edges)
      ? payload.edges.map(parsePlanEdge).filter((edge): edge is Stage4AdmissionPlanEdge => !!edge)
      : [],
    max_attempts: typeof payload.max_attempts === "number" ? payload.max_attempts : 4,
  };
}

function parseCheckResult(value: unknown): Stage4AdmissionCheckResult | null {
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

function parseReport(payload: Record<string, unknown>): Stage4AdmissionReport | null {
  if (typeof payload.name !== "string" || typeof payload.outcome !== "string") return null;
  const coupledRecheck = isRecord(payload.coupled_recheck)
    ? {
        constructs: stringArray(payload.coupled_recheck.constructs),
        closing_edges: stringArray(payload.coupled_recheck.closing_edges),
        results: Array.isArray(payload.coupled_recheck.results)
          ? payload.coupled_recheck.results
              .map(parseCheckResult)
              .filter((result): result is Stage4AdmissionCheckResult => result !== null)
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
          .filter((result): result is Stage4AdmissionCheckResult => result !== null)
      : [],
    parameters: Array.isArray(payload.parameters) ? parseParameters(payload.parameters) : undefined,
    coupled_recheck:
      coupledRecheck && coupledRecheck.results.length > 0 ? coupledRecheck : undefined,
  };
}

export function parseStage4AdmissionEvent(
  record: Stage4AdmissionEventRecord | null | undefined,
): Stage4AdmissionEvent | null {
  if (!record?.event?.startsWith(STAGE4_ADMISSION_EVENT_PREFIX) || !record.payload) return null;
  const eventType = record.event.slice(STAGE4_ADMISSION_EVENT_PREFIX.length);
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
      message: typeof payload.message === "string" ? payload.message : "Stage 4 failed",
    };
  }

  if (eventType === "done") return { type: "done" };
  return null;
}

function statusFromReport(report: Stage4AdmissionReport): Stage4AdmissionConstructStatus {
  if (!report.admitted) return "revising";
  return report.annotations.length > 0 ? "admitted_with_consequences" : "admitted";
}

function updateConstruct(
  constructs: Stage4AdmissionConstructState[],
  name: string,
  update: (construct: Stage4AdmissionConstructState) => Stage4AdmissionConstructState,
): Stage4AdmissionConstructState[] {
  return constructs.map((construct) => (construct.name === name ? update(construct) : construct));
}

export function applyStage4AdmissionEvent(
  state: Stage4AdmissionReplayState | undefined,
  event: Stage4AdmissionEvent,
): Stage4AdmissionReplayState {
  const current = state ?? EMPTY_STAGE4_ADMISSION_REPLAY_STATE;

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
