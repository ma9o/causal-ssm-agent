import { afterEach, describe, expect, it, vi } from "vitest";
import type { EpisodeArtifactId } from "@/lib/server/artifacts";

vi.mock("@/lib/storage", () => ({
  readData: vi.fn(),
  readBinary: vi.fn(),
  LOCAL_DATA_DIR: "/tmp/data",
  isStorageNotFoundError: (error: unknown) =>
    error instanceof Error && "code" in error && error.code === "ENOENT",
}));

vi.mock("@/lib/raw-data", () => ({
  deriveRawDataData: vi.fn(() => ({ marker: "raw_data-derived" })),
}));

vi.mock("@/lib/measurements-data", () => ({
  deriveMeasurementsData: vi.fn(() => ({ marker: "measurements-derived" })),
}));

vi.mock("@/lib/model-spec-derived-data", () => ({
  deriveStatisticalModelSpecData: vi.fn(() => ({ marker: "statistical_model_spec-derived" })),
}));

import { deriveRawDataData } from "@/lib/raw-data";
import { deriveMeasurementsData } from "@/lib/measurements-data";
import { deriveStatisticalModelSpecData } from "@/lib/model-spec-derived-data";
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

describe("GET /api/artifacts/[workspaceId]/[artifactId]/view", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("normalizes payloads that contain non-finite numbers instead of returning a fake 404", async () => {
    mockJsonFiles({
      "user/episode/state.json": stateFor("posterior"),
      [artifactPath("user", "posterior", "diagnostics.json")]:
        '{"value":1e999,"upper":1e999,"lower":-1e999,"label":"Infinity should stay a string"}',
    });

    const response = await GET(new Request("http://localhost/api/artifacts/user/posterior/view"), {
      params: Promise.resolve({ workspaceId: "user", artifactId: "posterior" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      value: null,
      upper: null,
      lower: null,
      label: "Infinity should stay a string",
    });
  });

  it("normalizes top-level non-finite numbers in persisted artifact payloads", async () => {
    mockJsonFiles({
      "user/episode/state.json": stateFor("posterior"),
      [artifactPath("user", "posterior", "diagnostics.json")]:
        '{"outcome":"warn","inference_metadata":{"duration_seconds":1e999}}',
    });

    const response = await GET(new Request("http://localhost/api/artifacts/user/posterior/view"), {
      params: Promise.resolve({ workspaceId: "user", artifactId: "posterior" }),
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

    const response = await GET(new Request("http://localhost/api/artifacts/user/posterior/view"), {
      params: Promise.resolve({ workspaceId: "user", artifactId: "posterior" }),
    });

    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        error: expect.stringContaining("Failed to read posterior"),
      }),
    );
  });

  it("hydrates raw_data from the parquet artifact instead of trusting persisted convenience fields", async () => {
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

    const response = await GET(new Request("http://localhost/api/artifacts/user/raw_data/view"), {
      params: Promise.resolve({ workspaceId: "user", artifactId: "raw_data" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ marker: "raw_data-derived" });
    expect(deriveRawDataData).toHaveBeenCalledWith(persisted, rawParquet);
  });

  it("hydrates measurements from the raw parquet artifact instead of trusting persisted convenience fields", async () => {
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
      "user/episode/state.json": stateFor("measurements", "panel"),
      [artifactPath("user", "measurements", "measurements.json")]: persisted,
    });
    vi.mocked(readBinary).mockImplementation(async (path: string) => {
      if (path === artifactPath("user", "panel", "panel.parquet")) return parquet;
      throw storageMissing(path);
    });

    const response = await GET(
      new Request("http://localhost/api/artifacts/user/measurements/view"),
      {
        params: Promise.resolve({ workspaceId: "user", artifactId: "measurements" }),
      },
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ marker: "measurements-derived" });
    expect(deriveMeasurementsData).toHaveBeenCalledWith(persisted, parquet);
  });

  it("hydrates model-spec likelihood diagnostics from validation + full measurements observations", async () => {
    const modelSpec = {
      statistical_model_spec: {
        likelihoods: [],
        parameters: [],
      },
      authored_priors: {},
      resolved_priors: [],
    };
    const validationReport = {
      is_valid: true,
      indicators: {},
      dataset_issues: [],
    };
    const parquet = new Uint8Array([7, 8, 9]);

    mockJsonFiles({
      "user/episode/state.json": stateFor("statistical_model_spec", "validation_report", "panel"),
      [artifactPath("user", "statistical_model_spec", "statistical_model_spec.json")]: modelSpec,
      [artifactPath("user", "validation_report", "validation_report.json")]: validationReport,
    });
    vi.mocked(readBinary).mockImplementation(async (path: string) => {
      if (path === artifactPath("user", "panel", "panel.parquet")) return parquet;
      throw storageMissing(path);
    });

    const response = await GET(
      new Request("http://localhost/api/artifacts/user/statistical_model_spec/view"),
      {
        params: Promise.resolve({ workspaceId: "user", artifactId: "statistical_model_spec" }),
      },
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({ marker: "statistical_model_spec-derived" });
    expect(deriveStatisticalModelSpecData).toHaveBeenCalledWith(
      modelSpec,
      validationReport,
      parquet,
    );
  });
});
