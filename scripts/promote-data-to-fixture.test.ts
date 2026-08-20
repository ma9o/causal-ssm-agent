import { afterEach, describe, expect, it } from "bun:test";
import { access, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { promoteDataWorkspace } from "./promote-data-to-fixture";

const ARTIFACTS = [
  "question",
  "raw_data",
  "latent_structure",
  "measurement_structure",
  "causal_design",
  "structural_plan",
  "identification_report",
  "measurements",
  "panel",
  "validation_report",
  "statistical_model_spec",
  "compiled_ssm",
  "posterior",
  "baseline_report",
] as const;

const PAYLOADS = {
  raw_data: "profile.json",
  latent_structure: "latent-structure.json",
  measurement_structure: "measurement_structure.json",
  causal_design: "causal_design.json",
  structural_plan: "structural-plan.json",
  measurements: "measurements.json",
  validation_report: "validation_report.json",
  statistical_model_spec: "statistical_model_spec.json",
  posterior: "diagnostics.json",
  baseline_report: "baseline_report.json",
} as const;

const TRACE_IDS = {
  raw_data: "raw-data",
  latent_structure: "latent-structure",
  measurement_structure: "measurement-structure",
  measurements: "measurement-chunk-000000-attempt-001",
  statistical_model_spec: "model-spec-sleep-attempt-001",
  baseline_report: "baseline-report",
} as const;

const temporaryRoots: string[] = [];

async function pathExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(join(path, ".."), { recursive: true });
  await writeFile(path, JSON.stringify(value));
}

async function seedCompleteWorkspace(
  dataRoot: string,
  workspaceId: string,
  options: { omit?: string; staleBaseline?: boolean } = {},
): Promise<void> {
  const workspaceRoot = join(dataRoot, workspaceId);
  await mkdir(join(workspaceRoot, "input"), { recursive: true });
  await writeFile(join(workspaceRoot, "input", "bundle.zip"), "fixture input");
  await writeJson(join(workspaceRoot, "access.json"), { version: 1 });
  await writeJson(join(workspaceRoot, "scratch", "discard.json"), { discard: true });
  await writeJson(join(workspaceRoot, "cache", "discard.json"), { discard: true });

  for (const [index, artifactId] of ARTIFACTS.entries()) {
    if (artifactId === options.omit) continue;

    const seq = index + 1;
    const version = 2;
    const derivedFrom =
      artifactId === "baseline_report" && options.staleBaseline ? { posterior: 1 } : {};
    const info = {
      artifact_id: artifactId,
      version,
      provenance: "computed",
      derived_from: derivedFrom,
      produced_by: `run:${artifactId}`,
      created_at: "2026-08-07T00:00:00Z",
    };
    const versionRoot = join(workspaceRoot, "store", artifactId, `v${version}`);
    await writeJson(join(versionRoot, "meta.json"), info);

    if (artifactId in PAYLOADS) {
      const filename = PAYLOADS[artifactId as keyof typeof PAYLOADS];
      await writeJson(join(versionRoot, filename), { artifact: artifactId, version });
    }

    const traceId = TRACE_IDS[artifactId as keyof typeof TRACE_IDS];
    if (traceId) {
      await writeJson(
        join(workspaceRoot, "episode", "traces", String(seq).padStart(6, "0"), `${traceId}.json`),
        { artifact: artifactId, trace: traceId },
      );
    }

    await writeJson(
      join(workspaceRoot, "episode", "journal", `${String(seq).padStart(6, "0")}.json`),
      {
        seq,
        status: "applied",
        produced: [info],
        retracted: [],
        trace_ids: traceId ? [traceId] : [],
      },
    );
  }
}

afterEach(async () => {
  await Promise.all(
    temporaryRoots.splice(0).map((root) => rm(root, { recursive: true, force: true })),
  );
});

describe("promoteDataWorkspace", () => {
  it("replaces DEMO with one durable workspace and copy-only fixture projections", async () => {
    const root = await mkdtemp(join(tmpdir(), "nof1-fixture-promotion-"));
    temporaryRoots.push(root);
    const dataRoot = join(root, "data");
    await seedCompleteWorkspace(dataRoot, "CANDIDATE");

    await writeJson(join(dataRoot, "DEMO", "fixture", "artifacts", "artificial.json"), {
      artificial: true,
    });
    await writeJson(join(dataRoot, "DEMO", "scratch", "old.json"), { old: true });

    const summary = await promoteDataWorkspace({
      sourceWorkspaceId: "CANDIDATE",
      dataRoot,
    });

    expect(summary.artifacts).toHaveLength(10);
    expect(summary.traces).toHaveLength(6);
    expect(
      JSON.parse(
        await readFile(join(dataRoot, "DEMO", "fixture", "artifacts", "posterior.json"), "utf8"),
      ),
    ).toEqual({ artifact: "posterior", version: 2 });
    expect(
      JSON.parse(
        await readFile(
          join(dataRoot, "DEMO", "fixture", "traces", "statistical_model_spec.json"),
          "utf8",
        ),
      ),
    ).toEqual({ artifact: "statistical_model_spec", trace: "model-spec-sleep-attempt-001" });
    expect(await pathExists(join(dataRoot, "DEMO", "store", "posterior", "v2", "meta.json"))).toBe(
      true,
    );
    expect(
      await pathExists(join(dataRoot, "DEMO", "fixture", "artifacts", "artificial.json")),
    ).toBe(false);
    expect(await pathExists(join(dataRoot, "DEMO", "scratch"))).toBe(false);
    expect(await pathExists(join(dataRoot, "DEMO", "cache"))).toBe(false);
    expect(await pathExists(join(dataRoot, "DEMO", "access.json"))).toBe(true);
  });

  it("leaves the existing fixture untouched when the source run is incomplete", async () => {
    const root = await mkdtemp(join(tmpdir(), "nof1-fixture-promotion-"));
    temporaryRoots.push(root);
    const dataRoot = join(root, "data");
    await seedCompleteWorkspace(dataRoot, "CANDIDATE", { omit: "baseline_report" });
    await mkdir(join(dataRoot, "DEMO"), { recursive: true });
    await writeFile(join(dataRoot, "DEMO", "sentinel.txt"), "keep me");

    await expect(
      promoteDataWorkspace({ sourceWorkspaceId: "CANDIDATE", dataRoot }),
    ).rejects.toThrow("missing current artifacts: baseline_report");

    expect(await readFile(join(dataRoot, "DEMO", "sentinel.txt"), "utf8")).toBe("keep me");
    expect((await readdir(dataRoot)).some((entry) => entry.startsWith(".DEMO-promotion-"))).toBe(
      false,
    );
  });

  it("rejects a complete-looking source whose current provenance chain is stale", async () => {
    const root = await mkdtemp(join(tmpdir(), "nof1-fixture-promotion-"));
    temporaryRoots.push(root);
    const dataRoot = join(root, "data");
    await seedCompleteWorkspace(dataRoot, "CANDIDATE", { staleBaseline: true });
    await mkdir(join(dataRoot, "DEMO"), { recursive: true });
    await writeFile(join(dataRoot, "DEMO", "sentinel.txt"), "keep me");

    await expect(
      promoteDataWorkspace({ sourceWorkspaceId: "CANDIDATE", dataRoot }),
    ).rejects.toThrow("stale current artifacts: baseline_report");
    expect(await readFile(join(dataRoot, "DEMO", "sentinel.txt"), "utf8")).toBe("keep me");
  });
});
