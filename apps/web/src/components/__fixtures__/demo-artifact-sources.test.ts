import { existsSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { temporaryDemoRunArtifactTargets } from "./demo-artifact-sources";

const repoRoot = fileURLToPath(new URL("../../../../../", import.meta.url));

describe("temporary demo-run artifact fixtures", () => {
  it("only covers DEMO store artifacts that have not been materialized", () => {
    const materializedTargets = Object.entries(temporaryDemoRunArtifactTargets)
      .filter(([, target]) => existsSync(join(repoRoot, target)))
      .map(([artifactId, target]) => `${artifactId}: ${target}`);

    expect(
      materializedTargets,
      [
        "Replace the matching temporary demo-run fixture imports in demo-artifact-sources.ts.",
        ...materializedTargets,
      ].join("\n"),
    ).toEqual([]);
  });
});
