import {
  EMPTY_STAGE4_ADMISSION_REPLAY_STATE,
  STAGE4_ADMISSION_EVENT_PREFIX,
  applyStage4AdmissionEvent,
  parseStage4AdmissionEvent,
} from "./stage4-admission-runtime";
import { describe, expect, it } from "vitest";

function replayStage4AdmissionEvents(
  records: readonly Parameters<typeof parseStage4AdmissionEvent>[0][],
) {
  return records.reduce((state, record) => {
    const event = parseStage4AdmissionEvent(record);
    return event ? applyStage4AdmissionEvent(state, event) : state;
  }, EMPTY_STAGE4_ADMISSION_REPLAY_STATE);
}

describe("stage4-admission-runtime", () => {
  it("parses and reduces construct admission events", () => {
    const planRecord = {
      event: `${STAGE4_ADMISSION_EVENT_PREFIX}plan`,
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
      event: `${STAGE4_ADMISSION_EVENT_PREFIX}construct_started`,
      payload: { type: "construct_started", construct: "stress_load", attempt: 1 },
    };
    const reportRecord = {
      event: `${STAGE4_ADMISSION_EVENT_PREFIX}construct_report`,
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

    expect(parseStage4AdmissionEvent(planRecord)?.type).toBe("plan");
    expect(parseStage4AdmissionEvent(startedRecord)?.type).toBe("construct_started");
    const parsedReport = parseStage4AdmissionEvent(reportRecord);
    expect(parsedReport?.type).toBe("construct_report");
    if (parsedReport?.type !== "construct_report") {
      throw new Error("expected construct_report");
    }
    expect(parsedReport.report.coupled_recheck).toMatchObject({
      constructs: ["stress_load", "sleep_disturbance"],
      closing_edges: ["sleep_disturbance->stress_load"],
    });

    const state = replayStage4AdmissionEvents([planRecord, startedRecord, reportRecord]);
    expect(state.constructs[0]?.status).toBe("admitted");
    expect(state.constructs[0]?.attempt).toBe(1);
    expect(state.latestReport?.results[0]?.duration_ms).toBe(640);
    expect(state.latestReport?.name).toBe("stress_load");
    expect(state.latestReport?.coupled_recheck?.results[0]?.target).toBe("sleep_disturbance");
  });

  it("keeps non-admitted reports in revision state", () => {
    const state = applyStage4AdmissionEvent(EMPTY_STAGE4_ADMISSION_REPLAY_STATE, {
      type: "plan",
      plan: {
        max_attempts: 4,
        constructs: [{ name: "sleep_disturbance" }],
        edges: [],
      },
    });
    const next = applyStage4AdmissionEvent(state, {
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
