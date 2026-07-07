import { afterEach, describe, expect, it, vi } from "vitest";
import type { EpisodeArtifactId } from "@/lib/server/artifacts";

vi.mock("@/lib/storage", () => ({
  readData: vi.fn(),
  readBinary: vi.fn(),
  LOCAL_DATA_DIR: "/tmp/data",
  isStorageNotFoundError: (error: unknown) =>
    error instanceof Error && "code" in error && error.code === "ENOENT",
}));

vi.mock("@/lib/stage0-data", () => ({
  deriveStage0Data: vi.fn(() => ({ marker: "stage-0-derived" })),
}));

vi.mock("@/lib/stage2-data", () => ({
  deriveStage2Data: vi.fn(() => ({ marker: "stage-2-derived" })),
}));

vi.mock("@/lib/stage4-derived-data", () => ({
  deriveStage4Data: vi.fn(() => ({ marker: "stage-4-derived" })),
}));

import { deriveStage0Data } from "@/lib/stage0-data";
import { deriveStage2Data } from "@/lib/stage2-data";
import { deriveStage4Data } from "@/lib/stage4-derived-data";
import { readBinary, readData } from "@/lib/storage";
import { GET } from "./route";

function artifactInfo(artifactId: EpisodeArtifactId) {
  return {
    artifact_id: artifactId,
    version: 1,
    provenance: "computed",
    derived_from: {},
    produced_by: null,
    created_at: "",
  };
}

function stateFor(...artifactIds: EpisodeArtifactId[]) {
  return {
    current: Object.fromEntries(
      artifactIds.map((artifactId) => [artifactId, artifactInfo(artifactId)]),
    ),
  };
}

function artifactPath(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
  filename: string,
): string {
  return `${workspaceId}/store/${artifactId}/v1/${filename}`;
}

function storageMissing(path: string): Error & { code: string } {
  return Object.assign(new Error(`Missing test storage fixture: ${path}`), { code: "ENOENT" });
}

function mockJsonFiles(files: Record<string, unknown>): void {
  vi.mocked(readData).mockImplementation(async (path) => {
    if (!(path in files)) throw storageMissing(path);
    const value = files[path];
    return typeof value === "string" ? value : JSON.stringify(value);
  });
}

describe("GET /api/results/[workspaceId]/[stage]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("normalizes payloads that contain non-finite numbers instead of returning a fake 404", async () => {
    mockJsonFiles({
      "user/episode/state.json": stateFor("posterior"),
      [artifactPath("user", "posterior", "diagnostics.json")]:
        '{"value":1e999,"upper":1e999,"lower":-1e999,"label":"Infinity should stay a string"}',
    });

    const response = await GET(new Request("http://localhost/api/results/user/stage-5b"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5b" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      value: null,
      upper: null,
      lower: null,
      label: "Infinity should stay a string",
    });
  });

  it("normalizes top-level non-finite numbers in persisted stage payloads", async () => {
    mockJsonFiles({
      "user/episode/state.json": stateFor("posterior"),
      [artifactPath("user", "posterior", "diagnostics.json")]:
        '{"outcome":"warn","inference_metadata":{"duration_seconds":1e999}}',
    });

    const response = await GET(new Request("http://localhost/api/results/user/stage-5b"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5b" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      outcome: "warn",
      inference_metadata: { duration_seconds: null },
    });
  });

  it("returns a parse error when the persisted payload is invalid", async () => {
    mockJsonFiles({
      "user/episode/state.json": stateFor("posterior"),
      [artifactPath("user", "posterior", "diagnostics.json")]: '{"value":',
    });

    const response = await GET(new Request("http://localhost/api/results/user/stage-5b"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5b" }),
    });

    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        error: expect.stringContaining("Failed to read stage-5b"),
      }),
    );
  });

  it("hydrates stage 0 from the parquet artifact instead of trusting persisted convenience fields", async () => {
    const persisted = {
      n_records: 999,
      column_descriptions: [{ name: "timestamp", description: "Observation time" }],
    };
    const rawParquet = new Uint8Array([1, 2, 3]);

    mockJsonFiles({
      "user/episode/state.json": stateFor("raw_data"),
      [artifactPath("user", "raw_data", "profile.json")]: persisted,
    });
    vi.mocked(readBinary).mockImplementation(async (path: string) => {
      if (path === artifactPath("user", "raw_data", "raw.parquet")) return rawParquet;
      throw storageMissing(path);
    });

    const response = await GET(new Request("http://localhost/api/results/user/stage-0"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-0" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ marker: "stage-0-derived" });
    expect(deriveStage0Data).toHaveBeenCalledWith(persisted, rawParquet);
  });

  it("hydrates stage 2 from the raw parquet artifact instead of trusting persisted convenience fields", async () => {
    const persisted = {
      workers: [
        {
          worker_id: 0,
          status: "completed",
          n_windows: 1,
          n_extractions: 1,
        },
      ],
      per_indicator_counts: { poisoned: 999 },
      combined_extractions_sample: [{ indicator: "poisoned", value: "persisted-only" }],
    };
    const parquet = new Uint8Array([4, 5, 6]);

    mockJsonFiles({
      "user/episode/state.json": stateFor("extraction_report", "model_data"),
      [artifactPath("user", "extraction_report", "extraction_report.json")]: persisted,
    });
    vi.mocked(readBinary).mockImplementation(async (path: string) => {
      if (path === artifactPath("user", "model_data", "model_data.parquet")) return parquet;
      throw storageMissing(path);
    });

    const response = await GET(new Request("http://localhost/api/results/user/stage-2"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-2" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ marker: "stage-2-derived" });
    expect(deriveStage2Data).toHaveBeenCalledWith(persisted, parquet);
  });

  it("hydrates stage 4 likelihood diagnostics from stage 3 + full stage 2 observations", async () => {
    const stage4 = {
      statistical_model_spec: {
        likelihoods: [],
        parameters: [],
      },
      authored_priors: {},
      resolved_priors: [],
    };
    const stage3 = {
      is_valid: true,
      indicators: {},
      dataset_issues: [],
    };
    const parquet = new Uint8Array([7, 8, 9]);

    mockJsonFiles({
      "user/episode/state.json": stateFor("compiled_ssm", "validation_report", "model_data"),
      [artifactPath("user", "compiled_ssm", "report.json")]: stage4,
      [artifactPath("user", "validation_report", "validation_report.json")]: stage3,
    });
    vi.mocked(readBinary).mockImplementation(async (path: string) => {
      if (path === artifactPath("user", "model_data", "model_data.parquet")) return parquet;
      throw storageMissing(path);
    });

    const response = await GET(new Request("http://localhost/api/results/user/stage-4"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-4" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ marker: "stage-4-derived" });
    expect(deriveStage4Data).toHaveBeenCalledWith(stage4, stage3, parquet);
  });
});
