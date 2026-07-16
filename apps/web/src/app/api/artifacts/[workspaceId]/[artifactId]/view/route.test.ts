import { afterEach, describe, expect, it, vi } from "vitest";
import type { EpisodeArtifactId } from "@/lib/server/artifacts";

vi.mock("@/lib/raw-data", () => ({
  deriveRawDataData: vi.fn(() => ({ marker: "raw_data-derived" })),
}));

vi.mock("@/lib/measurements-data", () => ({
  deriveMeasurementsData: vi.fn(() => ({ marker: "measurements-derived" })),
}));

vi.mock("@/lib/model-spec-derived-data", () => ({
  deriveStatisticalModelSpecData: vi.fn(() => ({ marker: "statistical_model_spec-derived" })),
}));

import { deriveMeasurementsData } from "@/lib/measurements-data";
import { deriveStatisticalModelSpecData } from "@/lib/model-spec-derived-data";
import { deriveRawDataData } from "@/lib/raw-data";
import { GET } from "./route";

function artifactResponse(
  artifactId: EpisodeArtifactId,
  payload: Record<string, unknown>,
  binaryFiles: string[] = [],
) {
  return {
    workspace_id: "user",
    artifact_id: artifactId,
    version: 1,
    meta: {
      artifact_id: artifactId,
      version: 1,
      provenance: "computed",
      derived_from: {},
      produced_by: null,
      created_at: "",
    },
    payload,
    binary_files: binaryFiles,
  };
}

function jsonFetchResponse(value: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => JSON.stringify(value),
    json: async () => value,
  } as Response;
}

function binaryFetchResponse(value: Uint8Array, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    text: async () => "",
    arrayBuffer: async () => value.buffer.slice(value.byteOffset, value.byteOffset + value.length),
  } as Response;
}

function facadeArtifactPath(workspaceId: string, artifactId: EpisodeArtifactId): string {
  return `/api/episodes/${workspaceId}/artifacts/${artifactId}`;
}

function facadeFilePath(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
  filename: string,
): string {
  return `${facadeArtifactPath(workspaceId, artifactId)}/files/${filename}`;
}

function mockFacade(options: {
  artifacts?: Record<string, unknown>;
  files?: Record<string, Uint8Array>;
}): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const path = new URL(String(input)).pathname;
      if (options.artifacts && path in options.artifacts) {
        return jsonFetchResponse(options.artifacts[path]);
      }
      if (options.files && path in options.files) {
        return binaryFetchResponse(options.files[path]);
      }
      return jsonFetchResponse({ detail: `missing ${path}` }, 404);
    }),
  );
}

describe("GET /api/artifacts/[workspaceId]/[artifactId]/view", () => {
  afterEach(() => {
    vi.clearAllMocks();
    vi.unstubAllGlobals();
  });

  it("normalizes facade payloads that contain non-finite numbers instead of returning a fake 404", async () => {
    mockFacade({
      artifacts: {
        [facadeArtifactPath("user", "posterior")]: artifactResponse("posterior", {
          "diagnostics.json": {
            value: Infinity,
            upper: Infinity,
            lower: -Infinity,
            label: "Infinity should stay a string",
          },
        }),
      },
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

  it("surfaces facade failures as read errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonFetchResponse({ detail: "facade failed" }, 500)),
    );

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

    mockFacade({
      artifacts: {
        [facadeArtifactPath("user", "raw_data")]: artifactResponse(
          "raw_data",
          { "profile.json": persisted },
          ["raw.parquet"],
        ),
      },
      files: {
        [facadeFilePath("user", "raw_data", "raw.parquet")]: rawParquet,
      },
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

    mockFacade({
      artifacts: {
        [facadeArtifactPath("user", "measurements")]: artifactResponse("measurements", {
          "measurements.json": persisted,
        }),
      },
      files: {
        [facadeFilePath("user", "panel", "panel.parquet")]: parquet,
      },
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

  it("unwraps the causal design artifact for the measurement-structure view", async () => {
    const measurementStructure = {
      measurement_structure: { model_clock: "1d", indicators: [] },
    };
    const causalDesign = {
      latent: { constructs: [], edges: [] },
      measurement: { model_clock: "1d", indicators: [] },
      identifiability: {
        identifiable_treatments: {},
        non_identifiable_treatments: {},
      },
      estimation: {
        state_order: [],
        edges: [],
        induced_dependencies: [],
        known_inputs: [],
      },
    };

    mockFacade({
      artifacts: {
        [facadeArtifactPath("user", "measurement_structure")]: artifactResponse(
          "measurement_structure",
          { "measurement_structure.json": measurementStructure },
        ),
        [facadeArtifactPath("user", "causal_design")]: artifactResponse("causal_design", {
          "causal_design.json": { causal_design: causalDesign },
        }),
      },
    });

    const response = await GET(
      new Request("http://localhost/api/artifacts/user/measurement_structure/view"),
      {
        params: Promise.resolve({
          workspaceId: "user",
          artifactId: "measurement_structure",
        }),
      },
    );

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      ...measurementStructure,
      causal_design: causalDesign,
    });
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

    mockFacade({
      artifacts: {
        [facadeArtifactPath("user", "statistical_model_spec")]: artifactResponse(
          "statistical_model_spec",
          { "statistical_model_spec.json": modelSpec },
        ),
        [facadeArtifactPath("user", "validation_report")]: artifactResponse("validation_report", {
          "validation_report.json": validationReport,
        }),
      },
      files: {
        [facadeFilePath("user", "panel", "panel.parquet")]: parquet,
      },
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
