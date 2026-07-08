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
              { name: "rho_stress_load", distribution: "Beta", params: { alpha: 2, beta: 2 } },
            ],
          },
        ],
        edges: [],
      },
    };
    const startedRecord = {
      event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_started`,
      payload: { type: "construct_started", construct: "stress_load", attempt: 1 },
    };
    const reportRecord = {
      event: `${MODEL_SPEC_ADMISSION_EVENT_PREFIX}construct_report`,
      payload: {
        type: "construct_report",
        name: "stress_load",
        attempt: 1,
        outcome: "ADMITTED",
        admitted: true,
        annotations: [],
        results: [
          {
            check: "C1a finiteness",
            target: "stress_load",
            value: "nonfinite 0.0%",
            band: "0%",
            duration_ms: 640,
            passed: true,
            note: "",
            mode: "hard",
          },
        ],
        coupled_recheck: {
          constructs: ["stress_load", "sleep_disturbance"],
          closing_edges: ["sleep_disturbance->stress_load"],
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

    const state = replayModelSpecAdmissionEvents([planRecord, startedRecord, reportRecord]);
    expect(state.constructs[0]?.status).toBe("admitted");
    expect(state.constructs[0]?.attempt).toBe(1);
    expect(state.latestReport?.results[0]?.duration_ms).toBe(640);
    expect(state.latestReport?.name).toBe("stress_load");
    expect(state.latestReport?.coupled_recheck?.results[0]?.target).toBe("sleep_disturbance");
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
      },
    });

    expect(next.activeConstruct).toBe("sleep_disturbance");
    expect(next.constructs[0]?.status).toBe("revising");
  });
});
