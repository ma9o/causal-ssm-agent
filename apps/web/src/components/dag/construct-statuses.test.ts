import type { MeasurementStructureViewData, StructuralPlan } from "@nof1-causal-lab/api-types";
import { describe, expect, it } from "vitest";
import { deriveConstructStatuses } from "./construct-statuses";

type CausalDesign = MeasurementStructureViewData["causal_design"];

describe("deriveConstructStatuses", () => {
  it("uses the established marginalized/blocking presentation for backend exclusions", () => {
    const design = {
      latent: {
        constructs: ["state", "latent_u", "context", "blocked"].map((name) => ({
          name,
          description: name,
          role: "endogenous",
          is_outcome: name === "state",
          temporal_status: "time_varying",
        })),
        edges: [],
      },
      measurement: { indicators: [], model_clock: "daily" },
      known_inputs: [],
      scientific_only_constructs: [{ construct: "context", reason: "Interpretation only" }],
      identifiability: {
        identifiable_treatments: {
          state: {
            method: "do_calculus",
            estimand: "P(y|do(x))",
            marginalized_confounders: ["latent_u"],
            instruments: [],
          },
        },
        non_identifiable_treatments: {
          blocked: { confounders: ["latent_u"] },
        },
      },
    } as unknown as CausalDesign;
    const structuralPlan = {
      semantics: {
        constructs: Object.fromEntries(
          design.latent.constructs.map((construct) => [`construct:${construct.name}`, construct]),
        ),
        edges: {},
        indicators: {},
        model_clock: "daily",
      },
      dispositions: [
        ["state", "retained_state"],
        ["latent_u", "marginalized"],
        ["context", "identification_only"],
        ["blocked", "retained_state"],
      ].map(([name, disposition]) => ({
        source_id: `construct:${name}`,
        source_kind: "construct",
        disposition,
        reason: String(disposition),
      })),
    } as unknown as StructuralPlan;

    expect(deriveConstructStatuses(design, structuralPlan)).toEqual({
      state: "observed",
      latent_u: "blocking",
      context: "marginalized",
      blocked: "blocking",
    });
  });
});
