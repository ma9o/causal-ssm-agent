import type { InferenceStructureResult } from "@causal-ssm/api-types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { InferenceStructureCard } from "./inference-structure-card";

describe("InferenceStructureCard", () => {
  it("renders active path and first-pass split summaries", () => {
    const inferenceStructure: InferenceStructureResult = {
      likelihood_path: "composed",
      auto_method: "laplace_em",
      first_pass_rb: {
        status: "active",
        inactive_reason: null,
        latent_variables: [
          { name: "g0", method: "kalman" },
          { name: "s0", method: "particle" },
        ],
        obs_variables: [
          { name: "yg0", method: "kalman" },
          { name: "ys0", method: "particle" },
        ],
      },
    };

    const markup = renderToStaticMarkup(
      createElement(InferenceStructureCard, { inferenceStructure }),
    );

    expect(markup).toContain("Inference Structure");
    expect(markup).toContain("Kalman + Particle");
    expect(markup).toContain("Laplace-EM");
    expect(markup).toContain("Latents");
    expect(markup).toContain("Observed channels");
    expect(markup).toContain("Kalman-side");
    expect(markup).toContain("Particle-side");
  });
});
