import { beforeAll, describe, expect, test } from "bun:test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Point RESULTS_DIR at test fixtures before importing the module
process.env.RESULTS_DIR = resolve(__dirname, "fixtures");

const { readStageResult } = await import("../services/stage-reader");

describe("readStageResult", () => {
  test("returns null for nonexistent run", async () => {
    const result = await readStageResult("nonexistent-run", "stage-0");
    expect(result).toBeNull();
  });

  test("returns null for nonexistent stage", async () => {
    const result = await readStageResult("test-run", "stage-4b");
    expect(result).toBeNull();
  });

  test("reads stage-0 data", async () => {
    const result = await readStageResult("test-run", "stage-0");
    expect(result).not.toBeNull();
    expect(result!.outcome).toBe("success");
    expect(result!.n_records).toBe(100);
  });

  test("reads stage-1a data", async () => {
    const result = await readStageResult("test-run", "stage-1a");
    expect(result).not.toBeNull();
    expect(result!.outcome_name).toBe("sleep");
  });
});

describe("large array stripping — stage-5", () => {
  test("strips posterior_marginals and posterior_pairs by default", async () => {
    const result = await readStageResult("test-run", "stage-5");
    expect(result).not.toBeNull();

    // These should be replaced with "[omitted — N items]"
    expect(result!.posterior_marginals).toBeString();
    expect(result!.posterior_marginals).toContain("omitted");
    expect(result!.posterior_pairs).toBeString();
    expect(result!.posterior_pairs).toContain("omitted");

    // Non-large fields should be untouched
    expect(result!.outcome).toBe("success");
    expect(result!.inference_metadata).toEqual({
      method: "svi",
      n_samples: 1000,
      duration_seconds: 12.5,
    });
  });

  test("preserves large arrays when include_large_arrays=true", async () => {
    const result = await readStageResult("test-run", "stage-5", true);
    expect(result).not.toBeNull();

    expect(Array.isArray(result!.posterior_marginals)).toBe(true);
    expect(Array.isArray(result!.posterior_pairs)).toBe(true);
  });
});

describe("large array stripping — stage-6 (nested)", () => {
  test("strips posterior_draws from each intervention_result", async () => {
    const result = await readStageResult("test-run", "stage-6");
    expect(result).not.toBeNull();

    const results = result!.intervention_results as Array<Record<string, unknown>>;
    expect(results).toHaveLength(2);

    // Scalar fields should be preserved
    expect(results[0].treatment).toBe("stress");
    expect(results[0].effect_size).toBe(0.312);
    expect(results[0].prob_positive).toBe(0.978);

    // posterior_draws should be stripped
    expect(results[0].posterior_draws).toBeString();
    expect(results[0].posterior_draws as string).toContain("omitted");
    expect(results[0].posterior_draws as string).toContain("12 draws");

    expect(results[1].posterior_draws).toBeString();
    expect(results[1].posterior_draws as string).toContain("5 draws");
  });

  test("preserves nested arrays when include_large_arrays=true", async () => {
    const result = await readStageResult("test-run", "stage-6", true);
    expect(result).not.toBeNull();

    const results = result!.intervention_results as Array<Record<string, unknown>>;
    expect(Array.isArray(results[0].posterior_draws)).toBe(true);
    expect((results[0].posterior_draws as number[]).length).toBe(12);
  });
});
