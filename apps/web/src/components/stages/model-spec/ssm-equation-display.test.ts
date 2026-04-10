import type { ParameterSpec, PriorProposal } from "@causal-ssm/api-types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { SSMEquationDisplay } from "./ssm-equation-display";

const parameters: ParameterSpec[] = [
  {
    name: "rho_sleep",
    role: "ar_coefficient",
    constraint: "unit_interval",
    description: "Persistence for sleep.",
  },
  {
    name: "sigma_sleep",
    role: "residual_sd",
    constraint: "positive",
    description: "Innovation scale for sleep.",
  },
];

const priors: PriorProposal[] = [
  {
    parameter: "rho_sleep",
    distribution: "Beta",
    params: { alpha: 2, beta: 2 },
    sources: [],
    reasoning: "Daily persistence prior.",
  },
];

describe("SSMEquationDisplay", () => {
  it("marks missing semantic priors as not authored", () => {
    const markup = renderToStaticMarkup(
      createElement(SSMEquationDisplay, {
        likelihoods: [],
        parameters,
        priors,
      }),
    );

    expect(markup).toContain("Not authored");
    expect(markup).toContain(String.raw`\mu_{0,\,\text{sleep}}:\ \text{Not authored}`);
    expect(markup).toContain(String.raw`\sigma_{0,\,\text{sleep}}:\ \text{Not authored}`);
  });
});
