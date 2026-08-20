#!/usr/bin/env bun

import { randomUUID } from "node:crypto";
import { access, cp, mkdir, mkdtemp, readdir, readFile, rename, rm } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

const WORKSPACE_ID = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

const DURABLE_ENTRIES = [
  "access.json",
  "episode",
  "input",
  "query.txt",
  "session.json",
  "sources",
  "store",
] as const;

const COMPLETE_ARTIFACTS = [
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

const ARTIFACT_PROJECTIONS = {
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

const TRACE_PROJECTIONS = {
  raw_data: /^raw-data$/,
  latent_structure: /^latent-structure$/,
  measurement_structure: /^measurement-structure$/,
  measurements: /^measurement-chunk-/,
  statistical_model_spec: /^model-spec-/,
  baseline_report: /^baseline-report$/,
} as const;

type ProjectedArtifactId = keyof typeof ARTIFACT_PROJECTIONS;
type ProjectedTraceId = keyof typeof TRACE_PROJECTIONS;

interface ArtifactVersionInfo {
  artifact_id: string;
  version: number;
  derived_from?: Record<string, number>;
}

interface TransitionRecord {
  seq: number;
  status: string;
  produced?: ArtifactVersionInfo[];
  retracted?: Array<{ artifact_id: string }>;
  trace_ids?: string[];
}

interface CurrentArtifact {
  info: ArtifactVersionInfo;
  record: TransitionRecord;
}

interface PromotionOptions {
  sourceWorkspaceId: string;
  dataRoot?: string;
  fixtureWorkspaceId?: string;
}

export interface PromotionSummary {
  source: string;
  destination: string;
  artifacts: ProjectedArtifactId[];
  traces: ProjectedTraceId[];
}

async function exists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

function assertWorkspaceId(value: string, label: string): void {
  if (!WORKSPACE_ID.test(value)) {
    throw new Error(`${label} must match ${WORKSPACE_ID}; received ${JSON.stringify(value)}.`);
  }
}

async function readJournal(workspaceRoot: string): Promise<TransitionRecord[]> {
  const journalRoot = join(workspaceRoot, "episode", "journal");
  if (!(await exists(journalRoot))) {
    throw new Error(`Source workspace has no episode journal: ${journalRoot}`);
  }

  const filenames = (await readdir(journalRoot))
    .filter((filename) => /^\d{6}\.json$/.test(filename))
    .sort();
  if (filenames.length === 0) {
    throw new Error(`Source workspace has an empty episode journal: ${journalRoot}`);
  }

  return Promise.all(
    filenames.map(async (filename) => {
      const raw = await readFile(join(journalRoot, filename), "utf8");
      const record = JSON.parse(raw) as TransitionRecord;
      if (!Number.isInteger(record.seq) || record.seq < 1) {
        throw new Error(`Invalid transition sequence in ${join(journalRoot, filename)}.`);
      }
      return record;
    }),
  );
}

function replayCurrentArtifacts(records: TransitionRecord[]): Map<string, CurrentArtifact> {
  const current = new Map<string, CurrentArtifact>();

  for (const record of records) {
    if (record.status !== "applied") continue;

    for (const info of record.produced ?? []) {
      current.set(info.artifact_id, { info, record });
    }
    for (const retraction of record.retracted ?? []) {
      current.delete(retraction.artifact_id);
    }
  }

  return current;
}

function isStale(
  artifactId: string,
  current: Map<string, CurrentArtifact>,
  visiting = new Set<string>(),
): boolean {
  const artifact = current.get(artifactId);
  if (!artifact || visiting.has(artifactId)) return false;

  const nextVisiting = new Set(visiting).add(artifactId);
  for (const [inputId, pinnedVersion] of Object.entries(artifact.info.derived_from ?? {})) {
    const input = current.get(inputId);
    if (!input || input.info.version !== pinnedVersion) return true;
    if (isStale(inputId, current, nextVisiting)) return true;
  }
  return false;
}

async function validateCompleteWorkspace(
  workspaceRoot: string,
): Promise<Map<string, CurrentArtifact>> {
  for (const requiredEntry of ["episode", "input", "store"] as const) {
    const path = join(workspaceRoot, requiredEntry);
    if (!(await exists(path))) {
      throw new Error(`Source workspace is missing required durable entry: ${path}`);
    }
  }

  const current = replayCurrentArtifacts(await readJournal(workspaceRoot));
  const missing = COMPLETE_ARTIFACTS.filter((artifactId) => !current.has(artifactId));
  if (missing.length > 0) {
    throw new Error(
      `Source workspace is incomplete; missing current artifacts: ${missing.join(", ")}.`,
    );
  }

  const stale = COMPLETE_ARTIFACTS.filter((artifactId) => isStale(artifactId, current));
  if (stale.length > 0) {
    throw new Error(`Source workspace has stale current artifacts: ${stale.join(", ")}.`);
  }

  for (const artifactId of COMPLETE_ARTIFACTS) {
    const artifact = current.get(artifactId) as CurrentArtifact;
    const metaPath = join(
      workspaceRoot,
      "store",
      artifactId,
      `v${artifact.info.version}`,
      "meta.json",
    );
    if (!(await exists(metaPath))) {
      throw new Error(`Current artifact ${artifactId} is missing its metadata file: ${metaPath}`);
    }

    const meta = JSON.parse(await readFile(metaPath, "utf8")) as ArtifactVersionInfo;
    if (meta.artifact_id !== artifactId || meta.version !== artifact.info.version) {
      throw new Error(`Artifact metadata does not match the episode journal: ${metaPath}`);
    }
  }

  return current;
}

async function copyDurableWorkspace(sourceRoot: string, stagingRoot: string): Promise<void> {
  for (const entry of DURABLE_ENTRIES) {
    const source = join(sourceRoot, entry);
    if (!(await exists(source))) continue;
    await cp(source, join(stagingRoot, entry), { recursive: true });
  }
}

async function copyArtifactProjections(
  sourceRoot: string,
  stagingRoot: string,
  current: Map<string, CurrentArtifact>,
): Promise<ProjectedArtifactId[]> {
  const destinationRoot = join(stagingRoot, "fixture", "artifacts");
  await mkdir(destinationRoot, { recursive: true });

  const copied: ProjectedArtifactId[] = [];
  for (const [artifactId, filename] of Object.entries(ARTIFACT_PROJECTIONS) as Array<
    [ProjectedArtifactId, string]
  >) {
    const artifact = current.get(artifactId) as CurrentArtifact;
    const source = join(sourceRoot, "store", artifactId, `v${artifact.info.version}`, filename);
    if (!(await exists(source))) {
      throw new Error(
        `Current artifact ${artifactId} is missing its projected JSON file: ${source}`,
      );
    }
    await cp(source, join(destinationRoot, `${artifactId}.json`));
    copied.push(artifactId);
  }
  return copied;
}

async function copyTraceProjections(
  sourceRoot: string,
  stagingRoot: string,
  current: Map<string, CurrentArtifact>,
): Promise<ProjectedTraceId[]> {
  const destinationRoot = join(stagingRoot, "fixture", "traces");
  await mkdir(destinationRoot, { recursive: true });

  const copied: ProjectedTraceId[] = [];
  for (const [artifactId, tracePattern] of Object.entries(TRACE_PROJECTIONS) as Array<
    [ProjectedTraceId, RegExp]
  >) {
    const record = (current.get(artifactId) as CurrentArtifact).record;
    const matchingTraceIds = (record.trace_ids ?? []).filter((traceId) =>
      tracePattern.test(traceId),
    );
    if (matchingTraceIds.length === 0) {
      throw new Error(
        `Current ${artifactId} transition has no trace matching ${tracePattern}: seq ${record.seq}.`,
      );
    }

    const traceId = matchingTraceIds.sort()[0];
    const source = join(
      sourceRoot,
      "episode",
      "traces",
      String(record.seq).padStart(6, "0"),
      `${traceId}.json`,
    );
    if (!(await exists(source))) {
      throw new Error(`Current ${artifactId} trace is missing: ${source}`);
    }
    await cp(source, join(destinationRoot, `${artifactId}.json`));
    copied.push(artifactId);
  }
  return copied;
}

async function replaceFixture(stagingRoot: string, fixtureRoot: string): Promise<void> {
  const backupRoot = `${fixtureRoot}.backup-${randomUUID()}`;
  const hadFixture = await exists(fixtureRoot);

  if (hadFixture) await rename(fixtureRoot, backupRoot);
  try {
    await rename(stagingRoot, fixtureRoot);
  } catch (error) {
    if (hadFixture) await rename(backupRoot, fixtureRoot);
    throw error;
  }

  if (hadFixture) await rm(backupRoot, { recursive: true, force: true });
}

export async function promoteDataWorkspace({
  sourceWorkspaceId,
  dataRoot = join(repoRoot, "data"),
  fixtureWorkspaceId = "DEMO",
}: PromotionOptions): Promise<PromotionSummary> {
  assertWorkspaceId(sourceWorkspaceId, "Source workspace id");
  assertWorkspaceId(fixtureWorkspaceId, "Fixture workspace id");
  if (sourceWorkspaceId === fixtureWorkspaceId) {
    throw new Error("Source workspace and fixture workspace must be different.");
  }

  const sourceRoot = join(dataRoot, sourceWorkspaceId);
  const fixtureRoot = join(dataRoot, fixtureWorkspaceId);
  if (!(await exists(sourceRoot))) {
    throw new Error(`Source workspace does not exist: ${sourceRoot}`);
  }

  const current = await validateCompleteWorkspace(sourceRoot);
  await mkdir(dataRoot, { recursive: true });
  const stagingRoot = await mkdtemp(join(dataRoot, `.${fixtureWorkspaceId}-promotion-`));

  try {
    await copyDurableWorkspace(sourceRoot, stagingRoot);
    const artifacts = await copyArtifactProjections(sourceRoot, stagingRoot, current);
    const traces = await copyTraceProjections(sourceRoot, stagingRoot, current);
    await replaceFixture(stagingRoot, fixtureRoot);
    return { source: sourceRoot, destination: fixtureRoot, artifacts, traces };
  } finally {
    if (await exists(stagingRoot)) {
      await rm(stagingRoot, { recursive: true, force: true });
    }
  }
}

function sourceWorkspaceFromArgs(args: string[]): string {
  if (args.length !== 2 || args[0] !== "--from") {
    throw new Error("Usage: bun run fixture:promote --from <workspace-id>");
  }
  return args[1];
}

if (import.meta.main) {
  try {
    const sourceWorkspaceId = sourceWorkspaceFromArgs(process.argv.slice(2));
    const result = await promoteDataWorkspace({ sourceWorkspaceId });
    console.log(`Promoted ${result.source} to ${result.destination}.`);
    console.log(
      `Copied ${result.artifacts.length} artifact projections and ${result.traces.length} trace projections.`,
    );
  } catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exitCode = 1;
  }
}
