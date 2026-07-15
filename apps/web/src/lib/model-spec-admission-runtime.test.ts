import {
  EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE,
  MODEL_SPEC_ADMISSION_EVENT_PREFIX,
  applyModelSpecAdmissionEvent,
  parseModelSpecAdmissionEvent,
} from "./model-spec-admission-runtime";
import { describe, expect, it } from "vitest";

function replayModelSpecAdmissionEvents(
  records: readonly Parameters<typeof parseModelSpecAdmissionEvent>[0][],
) {
  return records.reduce((state, record) => {
    const event = parseModelSpecAdmissionEvent(record);
    return event ? applyModelSpecAdmissionEvent(state, event) : state;
  }, EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE);
}

describe("model-spec-admission-runtime", () => {
  it("parses and reduces construct admission events", () => {
    const planRecord = {
      event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}plan`,
      payload: {
        type: "plan",
        max_attempts: 4,
        constructs: [
          {
            name: "stress_load",
            parents: [],
            indicators: ["journal_stress_rating"],
            parameters: [
              {
                name: "rho_stress_load",
                distribution: "Beta",
                params: { alpha: 2, beta: 2 },
              },
            ],
          },
        ],
        edges: [],
      },
    };
    const startedRecord = {
      event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_started`,
      payload: {
        type: "construct_started",
        construct: "stress_load",
        attempt: 1,
      },
    };
    const checkingRecord = {
      event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_checking`,
      occurred: "2026-07-11T10:00:00.000Z",
      payload: {
        type: "construct_checking",
        construct: "stress_load",
        attempt: 1,
      },
    };
    const reportRecord = {
      event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_report`,
      occurred: "2026-07-11T10:00:00.640Z",
      payload: {
        type: "construct_report",
        name: "stress_load",
        attempt: 1,
        outcome: "ADMITTED",
        admitted: true,
        annotations: [],
        timings: [
          {
            phase: "prior_predictive",
            label: "Exact prior-predictive simulation",
            duration_ms: 512,
            checks: [],
          },
          {
            phase: "c1_confinement",
            label: "C1 confinement",
            duration_ms: 128,
            checks: ["C1a finiteness"],
          },
        ],
        results: [
          {
            check: "C1a finiteness",
            target: "stress_load",
            value: "nonfinite 0.0%",
            band: "0%",
            passed: true,
            note: "",
            mode: "hard",
          },
        ],
        coupled_recheck: {
          constructs: ["stress_load", "sleep_disturbance"],
          closing_edges: ["sleep_disturbance->stress_load"],
          timings: [
            {
              phase: "prior_predictive",
              label: "Exact prior-predictive simulation",
              duration_ms: 300,
              checks: [],
            },
          ],
          results: [
            {
              check: "C2 latent scale",
              target: "sleep_disturbance",
              value: "median sd 1.2",
              band: "[0.33, 3.00]",
              passed: true,
              note: "",
              mode: "soft",
            },
          ],
        },
      },
    };

    expect(parseModelSpecAdmissionEvent(planRecord)?.type).toBe("plan");
    expect(parseModelSpecAdmissionEvent(startedRecord)?.type).toBe("construct_started");
    const parsedReport = parseModelSpecAdmissionEvent(reportRecord);
    expect(parsedReport?.type).toBe("construct_report");
    if (parsedReport?.type !== "construct_report") {
      throw new Error("expected construct_report");
    }
    expect(parsedReport.report.coupled_recheck).toMatchObject({
      constructs: ["stress_load", "sleep_disturbance"],
      closing_edges: ["sleep_disturbance->stress_load"],
    });
    expect(parsedReport.report.timings.map((timing) => timing.phase)).toEqual([
      "prior_predictive",
      "c1_confinement",
    ]);
    expect(parsedReport.report.coupled_recheck?.timings[0]?.duration_ms).toBe(300);

    const state = replayModelSpecAdmissionEvents([
      planRecord,
      startedRecord,
      checkingRecord,
      reportRecord,
    ]);
    expect(state.constructs[0]?.status).toBe("admitted");
    expect(state.constructs[0]?.attempt).toBe(1);
    expect(state.latestReport?.durationMs).toBe(640);
    expect(state.latestReport?.name).toBe("stress_load");
    expect(state.latestReport?.coupled_recheck?.results[0]?.target).toBe("sleep_disturbance");
  });

  it("numbers repeated submissions and derives their end-to-end check runtimes", () => {
    const state = replayModelSpecAdmissionEvents([
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}plan`,
        payload: {
          max_attempts: 4,
          constructs: [{ name: "stress_load" }],
          edges: [],
        },
      },
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_checking`,
        occurred: "2026-07-11T10:00:00.000Z",
        payload: { construct: "stress_load", attempt: 1 },
      },
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_report`,
        occurred: "2026-07-11T10:00:01.250Z",
        payload: {
          name: "stress_load",
          attempt: 1,
          outcome: "NEEDS DECISION",
          admitted: false,
          annotations: [],
          results: [],
        },
      },
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_checking`,
        occurred: "2026-07-11T10:00:02.000Z",
        payload: { construct: "stress_load", attempt: 1 },
      },
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_report`,
        occurred: "2026-07-11T10:00:04.000Z",
        payload: {
          name: "stress_load",
          attempt: 1,
          outcome: "ADMITTED",
          admitted: true,
          annotations: [],
          results: [],
        },
      },
    ]);

    expect(state.constructs[0]?.reports.map((report) => report.attempt)).toEqual([1, 2]);
    expect(state.constructs[0]?.reports.map((report) => report.durationMs)).toEqual([1250, 2000]);
    expect(state.constructs[0]?.attempt).toBe(2);
  });

  it("keeps non-admitted reports in revision state", () => {
    const state = applyModelSpecAdmissionEvent(EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE, {
      type: "plan",
      plan: {
        max_attempts: 4,
        constructs: [{ name: "sleep_disturbance" }],
        edges: [],
      },
    });
    const next = applyModelSpecAdmissionEvent(state, {
      type: "construct_report",
      report: {
        name: "sleep_disturbance",
        attempt: 1,
        outcome: "NEEDS DECISION - revise the fragment or accept the consequence",
        admitted: false,
        annotations: [],
        results: [],
        timings: [],
      },
    });

    expect(next.activeConstructs).toEqual(["sleep_disturbance"]);
    expect(next.constructs[0]?.status).toBe("revising");
  });

  it("restores retained constructs from a resumed checkpoint", () => {
    const plan = applyModelSpecAdmissionEvent(EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE, {
      type: "plan",
      plan: {
        max_attempts: 4,
        constructs: [{ name: "sleep" }, { name: "stress" }],
        edges: [],
      },
    });
    const resumed = applyModelSpecAdmissionEvent(plan, {
      type: "resumed",
      checkpointRef: "model-spec-checkpoint:workspace/run/checkpoint.json",
      sourceCheckpointRef: "model-spec-checkpoint:workspace/run/source.json",
      pinsChanged: true,
      retainedConstructs: ["sleep"],
      reopenedConstruct: "stress",
      reason: "stress no longer passes the scale check",
    });

    expect(resumed.constructs.map((construct) => construct.status)).toEqual([
      "admitted",
      "pending",
    ]);
    expect(resumed.done).toBe(false);
    expect(resumed.error).toBeNull();
    expect(resumed.resume).toMatchObject({
      pinsChanged: true,
      reopenedConstruct: "stress",
      reason: "stress no longer passes the scale check",
    });
  });

  it("clears construct attempt history when a fresh plan starts", () => {
    const firstPlan = applyModelSpecAdmissionEvent(EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE, {
      type: "plan",
      plan: {
        max_attempts: 4,
        constructs: [{ name: "sleep" }, { name: "stress" }],
        edges: [],
      },
    });
    const reported = applyModelSpecAdmissionEvent(firstPlan, {
      type: "construct_report",
      report: {
        name: "sleep",
        attempt: 1,
        outcome: "ADMITTED",
        admitted: true,
        annotations: [],
        results: [],
        timings: [],
      },
    });
    const replanned = applyModelSpecAdmissionEvent(reported, {
      type: "plan",
      plan: {
        max_attempts: 4,
        constructs: [{ name: "sleep" }, { name: "stress" }, { name: "mood" }],
        edges: [],
      },
    });

    expect(replanned.constructs[0]?.reports).toEqual([]);
    expect(replanned.constructs[0]?.attempt).toBe(0);
    expect(replanned.constructs[2]?.reports).toEqual([]);
    expect(replanned.done).toBe(false);
  });

  it("tracks parallel construct checks and their timers independently", () => {
    const state = replayModelSpecAdmissionEvents([
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}plan`,
        payload: {
          max_attempts: 4,
          constructs: [{ name: "sleep" }, { name: "stress" }],
          edges: [],
        },
      },
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_checking`,
        occurred: "2026-07-11T10:00:00.000Z",
        payload: { construct: "sleep", attempt: 1 },
      },
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_checking`,
        occurred: "2026-07-11T10:00:01.000Z",
        payload: { construct: "stress", attempt: 1 },
      },
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_report`,
        occurred: "2026-07-11T10:00:05.000Z",
        payload: { name: "sleep", outcome: "ADMITTED", admitted: true },
      },
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_report`,
        occurred: "2026-07-11T10:00:07.000Z",
        payload: { name: "stress", outcome: "ADMITTED", admitted: true },
      },
    ]);

    expect(state.activeConstructs).toEqual([]);
    expect(state.constructs[0]?.reports[0]?.durationMs).toBe(5000);
    expect(state.constructs[1]?.reports[0]?.durationMs).toBe(6000);
  });

  it("reopens the failed barrier frontier", () => {
    const state = replayModelSpecAdmissionEvents([
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}plan`,
        payload: {
          max_attempts: 4,
          constructs: [{ name: "sleep" }, { name: "stress" }, { name: "mood" }],
          edges: [],
        },
      },
      ...["sleep", "stress", "mood"].map((name) => ({
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_report`,
        payload: { name, outcome: "ADMITTED", admitted: true },
      })),
      {
        event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}barrier_report`,
        payload: {
          passed: false,
          failed_constructs: ["stress"],
          reopened_constructs: ["stress", "mood"],
        },
      },
    ]);

    expect(state.constructs.map((construct) => construct.status)).toEqual([
      "admitted",
      "revising",
      "pending",
    ]);
  });

  it("restores construct attempt history only when the new plan resumes a checkpoint", () => {
    const firstPlan = applyModelSpecAdmissionEvent(EMPTY_MODEL_SPEC_ADMISSION_REPLAY_STATE, {
      type: "plan",
      plan: {
        max_attempts: 4,
        constructs: [{ name: "sleep" }, { name: "stress" }],
        edges: [],
      },
    });
    const reported = applyModelSpecAdmissionEvent(firstPlan, {
      type: "construct_report",
      report: {
        name: "sleep",
        attempt: 1,
        outcome: "ADMITTED",
        admitted: true,
        annotations: [],
        results: [],
        timings: [],
      },
    });
    const replanned = applyModelSpecAdmissionEvent(reported, {
      type: "plan",
      plan: {
        max_attempts: 4,
        constructs: [{ name: "sleep" }, { name: "stress" }],
        edges: [],
      },
    });
    const resumed = applyModelSpecAdmissionEvent(replanned, {
      type: "resumed",
      checkpointRef: "model-spec-checkpoint:workspace/run/checkpoint.json",
      sourceCheckpointRef: "model-spec-checkpoint:workspace/run/source.json",
      pinsChanged: false,
      retainedConstructs: ["sleep"],
    });

    expect(resumed.constructs[0]?.reports).toHaveLength(1);
    expect(resumed.constructs[0]?.attempt).toBe(1);
    expect(resumed.constructs[0]?.status).toBe("admitted");
    expect(resumed.constructs[1]?.reports).toEqual([]);
  });
});
