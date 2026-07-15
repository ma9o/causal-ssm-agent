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
  passed: boolean;
  note: string;
  diagnosis?: string[];
  mode?: "hard" | "soft";
}

export interface ModelSpecAdmissionTiming {
  phase: string;
  label: string;
  duration_ms: number;
  checks: string[];
}

export interface ModelSpecAdmissionCoupledRecheck {
  constructs: string[];
  closing_edges?: string[];
  results: ModelSpecAdmissionCheckResult[];
  timings: ModelSpecAdmissionTiming[];
}

export interface ModelSpecAdmissionReport {
  name: string;
  attempt: number;
  /** End-to-end runtime from check start through the completed report. */
  durationMs?: number;
  outcome: string;
  admitted: boolean;
  annotations: string[];
  results: ModelSpecAdmissionCheckResult[];
  timings: ModelSpecAdmissionTiming[];
  /** Priors authored for this attempt; populates the construct's "Authored parameters" table. */
  parameters?: ModelSpecAdmissionParameter[];
  coupled_recheck?: ModelSpecAdmissionCoupledRecheck;
}

export interface ModelSpecAdmissionConstructState extends ModelSpecAdmissionPlanConstruct {
  status: ModelSpecAdmissionConstructStatus;
  attempt: number;
  reports: ModelSpecAdmissionReport[];
}

export interface ModelSpecAdmissionResumeState {
  checkpointRef: string;
  sourceCheckpointRef: string;
  pinsChanged: boolean;
  retainedConstructs: string[];
  reopenedConstruct?: string;
  reason?: string;
}

export interface ModelSpecAdmissionReplayState {
  plan: ModelSpecAdmissionPlan | null;
  constructs: ModelSpecAdmissionConstructState[];
  /** Previous campaign state held only until a following `resumed` event establishes lineage. */
  resumeCandidateConstructs: ModelSpecAdmissionConstructState[];
  activeConstructs: string[];
  activeAttempts: Record<string, number>;
  activeCheckStartedAtMs: Record<string, number>;
  phase: "planning" | "authoring" | "checking" | "done" | "failed";
  done: boolean;
  latestReport: ModelSpecAdmissionReport | null;
  error: string | null;
  resume: ModelSpecAdmissionResumeState | null;
}

export interface ModelSpecAdmissionEventRecord {
  event?: string | null;
  occurred?: string | null;
  payload?: Record<string, unknown>;
}

export type ModelSpecAdmissionEvent =
  | { type: "plan"; plan: ModelSpecAdmissionPlan }
  | {
      type: "resumed";
      checkpointRef: string;
      sourceCheckpointRef: string;
      pinsChanged: boolean;
      retainedConstructs: string[];
      reopenedConstruct?: string;
      reason?: string;
    }
  | { type: "construct_started"; construct: string; attempt: number }
  | {
      type: "construct_checking";
      construct: string;
      attempt: number;
      occurredAtMs?: number;
    }
  | {
      type: "construct_report";
      report: ModelSpecAdmissionReport;
      occurredAtMs?: number;
    }
  | {
      type: "barrier_report";
      passed: boolean;
      failedConstructs: string[];
      reopenedConstructs: string[];
    }
  | { type: "failed"; construct?: string; message: string }
  | { type: "done" };

export const EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE: ModelSpecAdmissionReplayState = {
  plan: null,
  constructs: [],
  resumeCandidateConstructs: [],
  activeConstructs: [],
  activeAttempts: {},
  activeCheckStartedAtMs: {},
  phase: "planning",
  done: false,
  latestReport: null,
  error: null,
  resume: null,
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
    passed: value.passed === true,
    note: typeof value.note === "string" ? value.note : "",
    diagnosis: stringArray(value.diagnosis),
    mode: value.mode === "hard" || value.mode === "soft" ? value.mode : undefined,
  };
}

function parseTiming(value: unknown): ModelSpecAdmissionTiming | null {
  if (
    !isRecord(value) ||
    typeof value.phase !== "string" ||
    typeof value.label !== "string" ||
    typeof value.duration_ms !== "number"
  ) {
    return null;
  }
  return {
    phase: value.phase,
    label: value.label,
    duration_ms: value.duration_ms,
    checks: stringArray(value.checks),
  };
}

function parseTimings(value: unknown): ModelSpecAdmissionTiming[] {
  return Array.isArray(value)
    ? value.map(parseTiming).filter((timing): timing is ModelSpecAdmissionTiming => timing !== null)
    : [];
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
        timings: parseTimings(payload.coupled_recheck.timings),
      }
    : undefined;
  return {
    name: payload.name,
    attempt: typeof payload.attempt === "number" ? payload.attempt : 1,
    durationMs: typeof payload.duration_ms === "number" ? payload.duration_ms : undefined,
    outcome: payload.outcome,
    admitted: payload.admitted === true,
    annotations: stringArray(payload.annotations),
    results: Array.isArray(payload.results)
      ? payload.results
          .map(parseCheckResult)
          .filter((result): result is ModelSpecAdmissionCheckResult => result !== null)
      : [],
    timings: parseTimings(payload.timings),
    parameters: Array.isArray(payload.parameters) ? parseParameters(payload.parameters) : undefined,
    coupled_recheck:
      coupledRecheck && coupledRecheck.results.length > 0 ? coupledRecheck : undefined,
  };
}

function parseOccurredAtMs(occurred: string | null | undefined): number | undefined {
  if (!occurred) return undefined;
  const timestamp = Date.parse(occurred);
  return Number.isFinite(timestamp) ? timestamp : undefined;
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
    eventType === "resumed" &&
    typeof payload.checkpoint_ref === "string" &&
    typeof payload.source_checkpoint_ref === "string"
  ) {
    return {
      type: "resumed",
      checkpointRef: payload.checkpoint_ref,
      sourceCheckpointRef: payload.source_checkpoint_ref,
      pinsChanged: payload.pins_changed === true,
      retainedConstructs: stringArray(payload.retained_constructs),
      reopenedConstruct:
        typeof payload.reopened_construct === "string" ? payload.reopened_construct : undefined,
      reason: typeof payload.reason === "string" ? payload.reason : undefined,
    };
  }

  if (eventType === "construct_started" && typeof payload.construct === "string") {
    return {
      type: "construct_started",
      construct: payload.construct,
      attempt: typeof payload.attempt === "number" ? payload.attempt : 1,
    };
  }

  if (eventType === "construct_checking" && typeof payload.construct === "string") {
    return {
      type: "construct_checking",
      construct: payload.construct,
      attempt: typeof payload.attempt === "number" ? payload.attempt : 1,
      occurredAtMs: parseOccurredAtMs(record.occurred),
    };
  }

  if (eventType === "construct_report") {
    const report = parseReport(payload);
    return report
      ? {
          type: "construct_report",
          report,
          occurredAtMs: parseOccurredAtMs(record.occurred),
        }
      : null;
  }

  if (eventType === "barrier_report") {
    return {
      type: "barrier_report",
      passed: payload.passed === true,
      failedConstructs: stringArray(payload.failed_constructs),
      reopenedConstructs: stringArray(payload.reopened_constructs),
    };
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
      resumeCandidateConstructs: current.constructs,
      phase: "planning",
      activeConstructs: [],
      activeAttempts: {},
      activeCheckStartedAtMs: {},
      done: false,
      latestReport: null,
      error: null,
      resume: null,
      constructs: event.plan.constructs.map((construct) => ({
        ...construct,
        status: "pending",
        attempt: 0,
        reports: [],
      })),
    };
  }

  if (event.type === "resumed") {
    const retained = new Set(event.retainedConstructs);
    const previousByName = new Map(
      current.resumeCandidateConstructs.map((construct) => [construct.name, construct]),
    );
    return {
      ...current,
      resumeCandidateConstructs: [],
      activeConstructs: [],
      activeAttempts: {},
      activeCheckStartedAtMs: {},
      phase: "planning",
      done: false,
      error: null,
      resume: {
        checkpointRef: event.checkpointRef,
        sourceCheckpointRef: event.sourceCheckpointRef,
        pinsChanged: event.pinsChanged,
        retainedConstructs: event.retainedConstructs,
        reopenedConstruct: event.reopenedConstruct,
        reason: event.reason,
      },
      constructs: current.constructs.map((construct) => {
        const previous = previousByName.get(construct.name);
        const reports = previous?.reports ?? [];
        const latest = reports[reports.length - 1];
        return {
          ...construct,
          attempt: previous?.attempt ?? 0,
          reports,
          status: retained.has(construct.name)
            ? latest?.admitted
              ? statusFromReport(latest)
              : "admitted"
            : "pending",
        };
      }),
    };
  }

  if (event.type === "construct_started") {
    const attempt =
      (current.constructs.find((construct) => construct.name === event.construct)?.reports.length ??
        0) + 1;
    return {
      ...current,
      resumeCandidateConstructs: [],
      activeConstructs: [...new Set([...current.activeConstructs, event.construct])],
      activeAttempts: { ...current.activeAttempts, [event.construct]: attempt },
      activeCheckStartedAtMs: Object.fromEntries(
        Object.entries(current.activeCheckStartedAtMs).filter(([name]) => name !== event.construct),
      ),
      phase: "authoring",
      constructs: updateConstruct(current.constructs, event.construct, (construct) => ({
        ...construct,
        status: "active",
        attempt,
      })),
    };
  }

  if (event.type === "construct_checking") {
    const attempt =
      (current.constructs.find((construct) => construct.name === event.construct)?.reports.length ??
        0) + 1;
    return {
      ...current,
      resumeCandidateConstructs: [],
      activeConstructs: [...new Set([...current.activeConstructs, event.construct])],
      activeAttempts: { ...current.activeAttempts, [event.construct]: attempt },
      activeCheckStartedAtMs:
        event.occurredAtMs === undefined
          ? current.activeCheckStartedAtMs
          : { ...current.activeCheckStartedAtMs, [event.construct]: event.occurredAtMs },
      phase: "checking",
      constructs: updateConstruct(current.constructs, event.construct, (construct) => ({
        ...construct,
        status: "checking",
        attempt,
      })),
    };
  }

  if (event.type === "construct_report") {
    const construct = current.constructs.find((candidate) => candidate.name === event.report.name);
    const attempt = (construct?.reports.length ?? 0) + 1;
    const checkStartedAtMs = current.activeCheckStartedAtMs[event.report.name];
    const measuredDurationMs =
      checkStartedAtMs !== undefined && event.occurredAtMs !== undefined
        ? Math.max(0, event.occurredAtMs - checkStartedAtMs)
        : undefined;
    const report = {
      ...event.report,
      attempt,
      durationMs: event.report.durationMs ?? measuredDurationMs,
    };
    const activeConstructs = report.admitted
      ? current.activeConstructs.filter((name) => name !== report.name)
      : [...new Set([...current.activeConstructs, report.name])];
    const activeAttempts = report.admitted
      ? Object.fromEntries(
          Object.entries(current.activeAttempts).filter(([name]) => name !== report.name),
        )
      : { ...current.activeAttempts, [report.name]: report.attempt };
    const activeCheckStartedAtMs = Object.fromEntries(
      Object.entries(current.activeCheckStartedAtMs).filter(([name]) => name !== report.name),
    );
    return {
      ...current,
      resumeCandidateConstructs: [],
      latestReport: report,
      activeConstructs,
      activeAttempts,
      activeCheckStartedAtMs,
      phase:
        !report.admitted || Object.keys(activeCheckStartedAtMs).length > 0
          ? "checking"
          : "authoring",
      constructs: updateConstruct(current.constructs, report.name, (construct) => ({
        ...construct,
        status: statusFromReport(report),
        attempt: report.attempt,
        reports: [...construct.reports, report],
        parameters: report.parameters ?? construct.parameters,
      })),
    };
  }

  if (event.type === "barrier_report") {
    const failed = new Set(event.failedConstructs);
    const reopened = new Set(event.reopenedConstructs);
    return {
      ...current,
      resumeCandidateConstructs: [],
      activeConstructs: [],
      activeAttempts: {},
      activeCheckStartedAtMs: {},
      phase: event.passed ? "checking" : "authoring",
      constructs: current.constructs.map((construct) => ({
        ...construct,
        status: failed.has(construct.name)
          ? "revising"
          : reopened.has(construct.name)
            ? "pending"
            : construct.status,
      })),
    };
  }

  if (event.type === "failed") {
    return {
      ...current,
      resumeCandidateConstructs: [],
      activeConstructs: event.construct
        ? [...new Set([...current.activeConstructs, event.construct])]
        : current.activeConstructs,
      activeCheckStartedAtMs: {},
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
    resumeCandidateConstructs: [],
    activeConstructs: [],
    activeAttempts: {},
    activeCheckStartedAtMs: {},
    phase: "done",
    done: true,
  };
}
