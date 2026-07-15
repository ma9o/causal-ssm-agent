import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { OutputSection } from "./output-section";

describe("OutputSection", () => {
  it("keeps supplied progression content visible after a transition fails", () => {
    const html = renderToStaticMarkup(
      createElement(OutputSection, {
        title: "Statistical model specification",
        status: "failed",
        errorMessage: "Construct admission stopped",
        runningContent: createElement("div", null, "Checkpointed construct progression"),
      }),
    );

    expect(html).toContain("Transition failed");
    expect(html).toContain("Construct admission stopped");
    expect(html).toContain("Checkpointed construct progression");
  });
});
