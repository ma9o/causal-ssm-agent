import { afterAll, beforeAll, describe, expect, mock, test } from "bun:test";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));

// Set env vars before any module imports
process.env.RESULTS_DIR = resolve(__dirname, "fixtures");
process.env.SESSIONS_PATH = resolve(__dirname, "fixtures", "sessions.json");

// Mock Prefect service — must be called before dynamic imports below
mock.module(resolve(__dirname, "../services/prefect"), () => ({
  getFlowRun: async (runId: string) => ({
    id: runId,
    state: { type: "COMPLETED", name: "Completed" },
    parameters: { query: "test question", user_id: "TEST01" },
  }),
  getTaskRuns: async () => [
    { id: "t1", name: "stage-0-flow", state: { type: "COMPLETED", name: "Completed" } },
    { id: "t2", name: "stage-1a-flow", state: { type: "COMPLETED", name: "Completed" } },
    { id: "t3", name: "stage-1b-flow-retry-1", state: { type: "RUNNING", name: "Running" } },
  ],
  getDeploymentId: async () => "test-deployment-id",
  triggerRun: async () => "new-test-run-id",
}));

// Mock file upload — don't actually copy files
mock.module(resolve(__dirname, "../services/file-upload"), () => ({
  uploadDataFile: async () => "/mocked/path/data.zip",
}));

// Dynamic imports so mocks take effect before module evaluation
const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
const { InMemoryTransport } = await import("@modelcontextprotocol/sdk/inMemory.js");
const { McpServer } = await import("@modelcontextprotocol/sdk/server/mcp.js");
const { INTERACTIVE_STAGES, STAGE_IDS } = await import("../generated/stage-config");
const { registerAnalyzeTool } = await import("../tools/analyze");
const { registerResultsTool } = await import("../tools/results");
const { registerRefineTool } = await import("../tools/refine");

let client: InstanceType<typeof Client>;
let server: InstanceType<typeof McpServer>;

beforeAll(async () => {
  server = new McpServer({ name: "test-causal", version: "0.0.1" });
  registerAnalyzeTool(server);
  registerResultsTool(server);
  registerRefineTool(server);

  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  client = new Client({ name: "test-client", version: "0.0.1" });

  await server.connect(serverTransport);
  await client.connect(clientTransport);
});

afterAll(async () => {
  await client.close();
  await server.close();
});

// ── Tool listing ──

describe("tools/list", () => {
  test("returns exactly 3 tools", async () => {
    const { tools } = await client.listTools();
    expect(tools).toHaveLength(3);
  });

  test("tool names are analyze, results, refine", async () => {
    const { tools } = await client.listTools();
    const names = tools.map((t) => t.name).sort();
    expect(names).toEqual(["analyze", "refine", "results"]);
  });

  test("analyze schema has required question and data_path", async () => {
    const { tools } = await client.listTools();
    const analyze = tools.find((t) => t.name === "analyze")!;
    const schema = analyze.inputSchema as { required?: string[]; properties?: Record<string, unknown> };
    expect(schema.required).toContain("question");
    expect(schema.required).toContain("data_path");
    expect(schema.required).not.toContain("override_gates");
  });

  test("results stage enum matches all stage IDs", async () => {
    const { tools } = await client.listTools();
    const results = tools.find((t) => t.name === "results")!;
    const schema = results.inputSchema as { properties?: Record<string, { enum?: string[] }> };
    const stageEnum = schema.properties?.stage?.enum;
    expect(stageEnum).toEqual([...STAGE_IDS]);
  });

  test("refine stage enum matches interactive stages only", async () => {
    const { tools } = await client.listTools();
    const refine = tools.find((t) => t.name === "refine")!;
    const schema = refine.inputSchema as { properties?: Record<string, { enum?: string[] }> };
    const stageEnum = schema.properties?.stage?.enum;
    expect(stageEnum).toEqual([...INTERACTIVE_STAGES]);
  });
});

// ── analyze tool ──

describe("analyze tool", () => {
  test("returns run_id and session_code", async () => {
    const result = await client.callTool({
      name: "analyze",
      arguments: { question: "Does stress affect sleep?", data_path: "/tmp/test.zip" },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(data.run_id).toBe("new-test-run-id");
    expect(data.session_code).toBeString();
    expect(data.session_code).toHaveLength(6);
  });
});

// ── results tool ──

describe("results tool", () => {
  test("returns overall status when no stage specified", async () => {
    const result = await client.callTool({
      name: "results",
      arguments: { run_id: "test-run" },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(data.status).toBe("completed");
    expect(data.completed_stages).toContain("stage-0");
    expect(data.completed_stages).toContain("stage-1a");
    expect(data.running_stages).toContain("stage-1b");
  });

  test("returns stage data when stage specified", async () => {
    const result = await client.callTool({
      name: "results",
      arguments: { run_id: "test-run", stage: "stage-0" },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(data.outcome).toBe("success");
    expect(data.n_records).toBe(100);
  });

  test("strips large arrays from stage-5 by default", async () => {
    const result = await client.callTool({
      name: "results",
      arguments: { run_id: "test-run", stage: "stage-5" },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(typeof data.posterior_marginals).toBe("string");
    expect(data.posterior_marginals).toContain("omitted");
    expect(data.inference_metadata.method).toBe("svi");
  });

  test("strips nested posterior_draws from stage-6", async () => {
    const result = await client.callTool({
      name: "results",
      arguments: { run_id: "test-run", stage: "stage-6" },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(data.intervention_results[0].effect_size).toBe(0.312);
    expect(typeof data.intervention_results[0].posterior_draws).toBe("string");
    expect(data.intervention_results[0].posterior_draws).toContain("omitted");
  });

  test("preserves large arrays when include_large_arrays=true", async () => {
    const result = await client.callTool({
      name: "results",
      arguments: { run_id: "test-run", stage: "stage-5", include_large_arrays: true },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(Array.isArray(data.posterior_marginals)).toBe(true);
  });

  test("returns not_available for missing stage", async () => {
    const result = await client.callTool({
      name: "results",
      arguments: { run_id: "test-run", stage: "stage-4b" },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(data.status).toBe("not_available");
  });
});

// ── refine tool ──

describe("refine tool", () => {
  test("merges edits and returns new run_id", async () => {
    const result = await client.callTool({
      name: "refine",
      arguments: {
        run_id: "test-run",
        stage: "stage-1a",
        edits: {
          treatments: ["stress", "exercise"],
        },
      },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(data.new_run_id).toBe("new-test-run-id");
    expect(data.edited_stage).toBe("stage-1a");
    expect(data.resumes_from).toBe("stage-1b");
  });

  test("returns error for stage with no data", async () => {
    const result = await client.callTool({
      name: "refine",
      arguments: {
        run_id: "test-run",
        stage: "stage-4",
        edits: {},
      },
    });
    const content = result.content as Array<{ type: string; text: string }>;
    const data = JSON.parse(content[0].text);
    expect(data.error).toContain("no results");
  });
});
