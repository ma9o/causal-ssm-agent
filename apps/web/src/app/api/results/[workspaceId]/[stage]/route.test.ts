import { readFile } from "node:fs/promises";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/workspace-access", () => ({
  requireWorkspaceAccess: vi.fn().mockImplementation(async (_request: Request, workspaceId: string) => ({
    ok: true,
    workspaceId,
  })),
}));

vi.mock("@/lib/storage", () => ({
  readData: vi.fn(),
  readBinary: vi.fn(),
  LOCAL_DATA_DIR: "/tmp/data",
}));

import { readBinary, readData } from "@/lib/storage";
import { requireWorkspaceAccess } from "@/lib/workspace-access";
import { GET } from "./route";

describe("GET /api/results/[workspaceId]/[stage]", () => {
  afterEach(() => {
    vi.clearAllMocks();
  });

  it("unwraps Prefect payloads that contain non-finite numbers instead of returning a fake 404", async () => {
    vi.mocked(readData).mockResolvedValue(
      JSON.stringify({
        metadata: { storage_key: "/tmp/data/user/run/stage-5a.json" },
        result:
          '{"value":NaN,"upper":Infinity,"lower":-Infinity,"label":"Infinity should stay a string"}',
      }),
    );

    const response = await GET(new Request("http://localhost/api/results/user/stage-5a"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5a" }),
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

  it("returns a parse error when the persisted payload is invalid", async () => {
    vi.mocked(readData).mockResolvedValue('{"metadata":{},"result":"{"}');

    const response = await GET(new Request("http://localhost/api/results/user/stage-5a"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-5a" }),
    });

    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        error: expect.stringContaining("Invalid persisted data for stage-5a"),
      }),
    );
  });

  it("hydrates stage 0 from the parquet artifact instead of trusting persisted convenience fields", async () => {
    const [stage0Json, parquetBytes] = await Promise.all([
      readFile(
        new URL(
          "../../../../../../../../data/MEDICAL_SEMANTICS/run/stage-0.json",
          import.meta.url,
        ),
        "utf-8",
      ),
      readFile(
        new URL(
          "../../../../../../../../data/MEDICAL_SEMANTICS/run/stage0-raw-input.parquet",
          import.meta.url,
        ),
      ),
    ]);
    const persisted = JSON.parse(stage0Json) as {
      outcome: "success" | "warn" | "fail";
      llm_trace?: unknown;
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
        n_records: 95,
        n_columns: 34,
        date_range: { start: "2025-03-03", end: "2025-03-31" },
        sample: expect.any(Array),
        column_descriptions: expect.arrayContaining([
          expect.objectContaining({
            name: "timestamp",
            dtype: "Datetime(time_unit='us', time_zone=None)",
            description: "UTC datetime of the event/observation",
          }),
        ]),
      }),
    );
  });

  it("hydrates stage 2 from the raw parquet artifact instead of trusting persisted convenience fields", async () => {
    const [stage2Json, parquetBytes] = await Promise.all([
      readFile(
        new URL(
          "../../../../../../../../data/MEDICAL_SEMANTICS/run/stage-2.json",
          import.meta.url,
        ),
        "utf-8",
      ),
      readFile(
        new URL(
          "../../../../../../../../data/MEDICAL_SEMANTICS/run/stage2-model-data.parquet",
          import.meta.url,
        ),
      ),
    ]);
    const persisted = JSON.parse(stage2Json) as {
      outcome: "success" | "warn" | "fail";
      llm_trace?: unknown;
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

    vi.mocked(readData).mockResolvedValue(
      JSON.stringify({
        outcome: persisted.outcome,
        llm_trace: persisted.llm_trace,
        workers: persisted.workers,
      }),
    );
    vi.mocked(readBinary).mockResolvedValue(
      new Uint8Array(parquetBytes.buffer, parquetBytes.byteOffset, parquetBytes.byteLength),
    );

    const response = await GET(new Request("http://localhost/api/results/user/stage-2"), {
      params: Promise.resolve({ workspaceId: "user", stage: "stage-2" }),
    });

    expect(response.status).toBe(200);
    await expect(response.json()).resolves.toEqual(
      expect.objectContaining({
        outcome: "success",
        workers: persisted.workers,
        per_indicator_counts: persisted.per_indicator_counts,
        combined_extractions_sample: persisted.combined_extractions_sample,
      }),
    );
  });
});
