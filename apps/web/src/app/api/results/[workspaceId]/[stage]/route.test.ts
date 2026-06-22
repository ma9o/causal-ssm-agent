import { readFile } from "node:fs/promises";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { LLMTrace, Stage3Data, Stage4PersistedData } from "@nof1-causal-lab/api-types";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi
    .fn()
    .mockImplementation(async (_request: Request, workspaceId: string) => ({
      ok: true,
      workspaceId,
    })),
}));

vi.mock("@/lib/storage", () => ({
  readData: vi.fn(),
  readBinary: vi.fn(),
  LOCAL_DATA_DIR: "/tmp/data",
}));

import { deriveStage2Data } from "@/lib/stage2-data";
import { deriveStage4Data } from "@/lib/stage4-derived-data";
import { readBinary, readData } from "@/lib/storage";
import { SHARED_WORKSPACE_CACHE_CONTROL } from "@/lib/shared-workspace-cache";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { GET } from "./route";

describe("GET /api/results/[workspaceId]/[stage]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("unwraps Prefect payloads that contain non-finite numbers instead of returning a fake 404", async () => {
    vi.mocked(readData).mockResolvedValue(
      JSON.stringify({
        metadata: { storage_key: "/tmp/data/user/run/stage-5b.json" },
        result:
          '{"value":NaN,"upper":Infinity,"lower":-Infinity,"label":"Infinity should stay a string"}',
      }),
    );

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
    expect(requireWorkspaceAccess).toHaveBeenCalledWith(expect.any(Request), "user");
  });

  it("normalizes top-level non-finite numbers in persisted stage payloads", async () => {
    vi.mocked(readData).mockResolvedValue(
      '{"outcome":"warn","inference_metadata":{"duration_seconds":Infinity}}',
    );

    const response = await GET(new Request("http://localhost/api/results/user/stage-5b"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5b" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual({
      outcome: "warn",
      inference_metadata: { duration_seconds: null },
    });
  });

  it("adds public CDN caching to shared workspace result responses", async () => {
    vi.mocked(readData).mockResolvedValue('{"outcome":"success"}');

    const response = await GET(new Request("http://localhost/api/results/DEMO/stage-5b"), {
      params: Promise.resolve({ workspaceId: "DEMO", stage: "stage-5b" }),
    });

    expect(response.status).toBe(200);
    expect(response.headers.get("Cache-Control")).toBe(SHARED_WORKSPACE_CACHE_CONTROL);
  });

  it("keeps non-shared workspace result responses uncached", async () => {
    vi.mocked(readData).mockResolvedValue('{"outcome":"success"}');

    const response = await GET(new Request("http://localhost/api/results/user/stage-5b"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5b" }),
    });

    expect(response.status).toBe(200);
    expect(response.headers.get("Cache-Control")).toBeNull();
  });

  it("returns a parse error when the persisted payload is invalid", async () => {
    vi.mocked(readData).mockResolvedValue('{"metadata":{},"result":"{"}');

    const response = await GET(new Request("http://localhost/api/results/user/stage-5b"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5b" }),
    });

    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        error: expect.stringContaining("Invalid persisted data for stage-5b"),
      }),
    );
  });

  it("hydrates stage 0 from the parquet artifact instead of trusting persisted convenience fields", async () => {
    const [stage0Json, parquetBytes] = await Promise.all([
      readFile(
        new URL("../../../../../../../../data/DEMO/run/stage-0.json", import.meta.url),
        "utf-8",
      ),
      readFile(
        new URL("../../../../../../../../data/DEMO/run/stage0-raw-input.parquet", import.meta.url),
      ),
    ]);
    const persisted = JSON.parse(stage0Json) as {
      outcome: "success" | "warn" | "fail";
      llm_trace?: LLMTrace | null;
      column_descriptions: Array<{ name: string; description: string }>;
    };

    vi.mocked(readData).mockResolvedValue(
      JSON.stringify({
        outcome: persisted.outcome,
        llm_trace: persisted.llm_trace,
        column_descriptions: persisted.column_descriptions.map(({ name, description }) => ({
          name,
          description,
        })),
      }),
    );
    vi.mocked(readBinary).mockResolvedValue(
      new Uint8Array(parquetBytes.buffer, parquetBytes.byteOffset, parquetBytes.byteLength),
    );

    const response = await GET(new Request("http://localhost/api/results/user/stage-0"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-0" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        outcome: "success",
        n_records: 1588,
        n_columns: 56,
        date_range: { start: "2022-01-01", end: "2026-05-07" },
        sample: expect.any(Array),
        column_descriptions: expect.arrayContaining([
          expect.objectContaining({
            name: "timestamp",
            dtype: "Datetime(time_unit='us', time_zone=None)",
          }),
        ]),
      }),
    );
  });

  it("hydrates stage 2 from the raw parquet artifact instead of trusting persisted convenience fields", async () => {
    const [stage2Json, parquetBytes] = await Promise.all([
      readFile(
        new URL("../../../../../../../../data/DEMO/run/stage-2.json", import.meta.url),
        "utf-8",
      ),
      readFile(
        new URL("../../../../../../../../data/DEMO/run/stage2-model-data.parquet", import.meta.url),
      ),
    ]);
    const persisted = JSON.parse(stage2Json) as {
      outcome: "success" | "warn" | "fail";
      llm_trace?: LLMTrace | null;
      workers: Array<{
        worker_id: number;
        indicator: string;
        status: "completed" | "failed";
        n_windows: number;
        n_extractions: number;
        error?: string | null;
      }>;
      per_indicator_counts: Record<string, number>;
      combined_extractions_sample: Array<Record<string, unknown>>;
    };
    const parquet = new Uint8Array(
      parquetBytes.buffer,
      parquetBytes.byteOffset,
      parquetBytes.byteLength,
    );
    const expected = await deriveStage2Data(
      {
        outcome: persisted.outcome ?? "success",
        llm_trace: persisted.llm_trace,
        workers: persisted.workers,
      },
      parquet,
    );

    vi.mocked(readData).mockResolvedValue(
      JSON.stringify({
        outcome: persisted.outcome ?? "success",
        llm_trace: persisted.llm_trace,
        workers: persisted.workers,
        per_indicator_counts: { poisoned: 999 },
        combined_extractions_sample: [{ indicator: "poisoned", value: "persisted-only" }],
      }),
    );
    vi.mocked(readBinary).mockResolvedValue(parquet);

    const response = await GET(new Request("http://localhost/api/results/user/stage-2"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-2" }),
    });

    expect(response.status).toBe(200);
    const payload = await response.json();
    expect(payload).toEqual(expected);
    expect(payload.per_indicator_counts).not.toEqual({ poisoned: 999 });
    expect(payload.combined_extractions_sample).not.toEqual([
      { indicator: "poisoned", value: "persisted-only" },
    ]);
  });

  it("hydrates stage 4 likelihood diagnostics from stage 3 + full stage 2 observations", async () => {
    const [stage4Json, stage3Json, parquetBytes] = await Promise.all([
      readFile(
        new URL("../../../../../../../../data/DEMO/run/stage-4.json", import.meta.url),
        "utf-8",
      ),
      readFile(
        new URL("../../../../../../../../data/DEMO/run/stage-3.json", import.meta.url),
        "utf-8",
      ),
      readFile(
        new URL("../../../../../../../../data/DEMO/run/stage2-model-data.parquet", import.meta.url),
      ),
    ]);
    const persisted = JSON.parse(stage4Json) as Stage4PersistedData;
    const stage3 = JSON.parse(stage3Json) as Stage3Data;
    const parquet = new Uint8Array(
      parquetBytes.buffer,
      parquetBytes.byteOffset,
      parquetBytes.byteLength,
    );
    const expected = await deriveStage4Data(persisted, stage3, parquet);

    vi.mocked(readData).mockResolvedValueOnce(stage4Json).mockResolvedValueOnce(stage3Json);
    vi.mocked(readBinary).mockResolvedValue(parquet);

    const response = await GET(new Request("http://localhost/api/results/user/stage-4"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-4" }),
    });

    expect(response.status).toBe(200);
    const payload = await response.json();
    expect(payload).toEqual(expected);
    expect(payload.likelihood_diagnostics.total_sleep_hours.histogram.length).toBeGreaterThan(1);
  });
});
