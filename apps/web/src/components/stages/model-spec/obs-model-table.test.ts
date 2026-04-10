import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { ObsPriorList } from "./obs-model-table";

const gaussianLikelihood = {
  variable: "sleep",
  distribution: "gaussian",
  link: "identity",
  centered: false,
  reasoning: "",
  sources: [],
} as const;

const betaLikelihood = {
  variable: "appointment_attendance",
  distribution: "beta",
  link: "logit",
  centered: false,
  reasoning: "",
  sources: [],
} as const;

describe("ObsPriorList", () => {
  it("marks missing authored observation priors as not authored", () => {
    const markup = renderToStaticMarkup(
      createElement(ObsPriorList, { likelihood: gaussianLikelihood, terms: [] }),
    );

    expect(markup).toContain("Not authored");
  });

  it("marks missing expected observation terms as not authored", () => {
    const markup = renderToStaticMarkup(
      createElement(ObsPriorList, {
        likelihood: gaussianLikelihood,
        terms: [{ parameterName: "obs_sd_sleep" }],
      }),
    );

    expect(markup).toContain("Not authored");
    expect(markup).toContain("\\sigma_{\\text{sleep}}");
    expect(markup).not.toContain("obs sd sleep");
  });

  it("renders beta observation concentration as phi instead of a raw obs_* name", () => {
    const markup = renderToStaticMarkup(
      createElement(ObsPriorList, {
        likelihood: betaLikelihood,
        terms: [
          {
            parameterName: "obs_concentration",
            prior: {
              parameter: "obs_concentration",
              distribution: "Gamma",
              params: { alpha: 5, beta: 0.5 },
              sources: [],
              reasoning: "",
            },
          },
        ],
      }),
    );

    expect(markup).toContain("\\phi \\sim \\text{Gamma}(5,\\; 0.5)");
    expect(markup).not.toContain("obs concentration");
  });
});
