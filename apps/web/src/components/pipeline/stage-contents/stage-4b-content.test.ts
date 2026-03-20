import type { Stage4bData } from "@causal-ssm/api-types";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import Stage4bContent from "./stage-4b-content";

describe("Stage4bContent", () => {
  it("renders the T-rule card when Stage 4b includes a t_rule payload", () => {
    const data = {
      outcome: "warn",
      parametric_id: {
        checked: true,
        t_rule: {
          n_free_params: 12,
          n_moments: 10,
          satisfies: false,
          param_counts: {},
        },
        error:
          "T-rule warning: 12 free params > conservative lower-bound 10 moment conditions. This screen is warning-only and does not halt inference.",
      },
      inference_structure: null,
    } as Stage4bData;

    const markup = renderToStaticMarkup(createElement(Stage4bContent, { data }));

    expect(markup).toContain("T-Rule");
    expect(markup).toContain("Warning");
    expect(markup).toContain("Free params:");
    expect(markup).toContain("Lower-bound moments:");
    expect(markup).toContain("12");
    expect(markup).toContain("10");
  });
});
