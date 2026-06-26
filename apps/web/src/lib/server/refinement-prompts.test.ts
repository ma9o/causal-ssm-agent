import { describe, expect, it } from "vitest";

import { buildRefinementContextMessages } from "./refinement-prompts";

const stage4Data = {
  outcome: "success" as const,
  model_spec: {
    likelihoods: [
      {
        variable: "pss_score",
        distribution: "gaussian" as const,
        link: "identity" as const,
        reasoning: "Continuous score.",
        sources: [],
      },
    ],
    parameters: [
      {
        name: "beta_stress_sleep",
        role: "fixed_effect" as const,
        constraint: "none" as const,
        description: "Effect of stress on sleep.",
      },
    ],
  },
  authored_priors: {
    beta_stress_sleep: {
      parameter: "beta_stress_sleep",
      distribution: "Normal" as const,
      params: { mu: -0.2, sigma: 0.1 },
      sources: [],
      reasoning: "Prior from longitudinal literature.",
    },
  },
  resolved_priors: [],
  search_queries: {
    beta_stress_sleep: "daily stress sleep longitudinal effect size",
  },
};

describe("buildRefinementContextMessages", () => {
  it("returns a broad Stage 4 refinement prompt", () => {
    const messages = buildRefinementContextMessages("stage-4", stage4Data, {});

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({ role: "system" });
    expect(messages[1]).toMatchObject({ role: "user" });
    expect(String(messages[0].content)).toContain("live refinement path");
    expect(String(messages[1].content)).toContain(
      "All current Stage 4 decisions are shown together",
    );
    expect(String(messages[1].content)).toContain("## Your Decisions");
    expect(String(messages[1].content)).toContain("### 1. Likelihood Choices");
    expect(String(messages[1].content)).toContain("### 2. Loading Constraints");
    expect(String(messages[1].content)).toContain("### 3. Parameter Prior Cards");
    expect(String(messages[1].content)).toContain("beta_stress_sleep");
    expect(String(messages[1].content)).toContain("daily stress sleep longitudinal effect size");
    expect(String(messages[1].content)).not.toContain("## Full Current model_spec");
    expect(String(messages[1].content)).not.toContain("## Full Current authored_priors");
  });

  it("applies the pending patch to the rendered context", () => {
    const messages = buildRefinementContextMessages("stage-4", stage4Data, {
      authored_priors: {
        beta_stress_sleep: {
          parameter: "beta_stress_sleep",
          distribution: "Normal",
          params: { mu: -0.5, sigma: 0.2 },
          sources: [],
          reasoning: "Updated prior.",
        },
      },
    });

    expect(String(messages[1].content)).toContain(
      'current prior: `Normal` with {"mu":-0.5,"sigma":0.2}',
    );
    expect(String(messages[1].content)).toContain("`authored_priors`");
  });

  it("returns no synthetic context for non-stage-4 routes", () => {
    expect(buildRefinementContextMessages("stage-6", stage4Data, {})).toEqual([]);
  });
});
