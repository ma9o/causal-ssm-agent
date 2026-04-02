import { describe, expect, it } from "vitest";
import { summarizeValidationIssues } from "./indicator-health-table";

describe("summarizeValidationIssues", () => {
  it("counts actionable issues even when no check cells are populated", () => {
    expect(
      summarizeValidationIssues({
        issues: [
          {
            indicator: "sleep_quality",
            issue_type: "missing",
            severity: "warning",
            message: "No data extracted for this indicator",
          },
        ],
        checks: {},
      }),
    ).toEqual({ count: 1, hasError: false });
  });

  it("treats any error issue as a failing row", () => {
    expect(
      summarizeValidationIssues({
        issues: [
          {
            indicator: "sleep_quality",
            issue_type: "low_n",
            severity: "warning",
            message: "Only 3 observations remain",
          },
          {
            indicator: "sleep_quality",
            issue_type: "no_numeric",
            severity: "error",
            message: "No numeric values extracted",
          },
        ],
        checks: {},
      }),
    ).toEqual({ count: 2, hasError: true });
  });
});
