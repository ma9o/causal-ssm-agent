import { readBinary, readData } from "@/lib/storage";

export type EpisodeArtifactId =
  | "question"
  | "raw_data"
  | "constructs"
  | "causal_spec"
  | "identification_report"
  | "extraction_report"
  | "model_data"
  | "validation_report"
  | "compiled_ssm"
  | "posterior"
  | "baseline_ranking"
  | "saved_scenarios";

type FileKind = "json" | "parquet" | "pickle";

type ArtifactFileSpec = {
  json?: Record<string, string>;
  parquet?: Record<string, string>;
  pickle?: Record<string, string>;
};

type ArtifactVersionInfo = {
  artifact_id: EpisodeArtifactId;
  version: number;
  derived_from: Partial<Record<EpisodeArtifactId, number>>;
};

type EpisodeState = {
  current?: Partial<Record<EpisodeArtifactId, ArtifactVersionInfo>>;
};

export class ArtifactNotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ArtifactNotFoundError";
  }
}

export const ARTIFACT_FILE_SPECS: Record<EpisodeArtifactId, ArtifactFileSpec> = {
  question: { json: { question: "question.json" } },
  raw_data: { json: { profile: "profile.json" }, parquet: { raw: "raw.parquet" } },
  constructs: { json: { constructs: "constructs.json" } },
  causal_spec: { json: { causal_spec: "causal_spec.json" } },
  identification_report: { json: { identification_report: "identification_report.json" } },
  extraction_report: { json: { extraction_report: "extraction_report.json" } },
  model_data: { parquet: { model_data: "model_data.parquet" } },
  validation_report: { json: { validation_report: "validation_report.json" } },
  compiled_ssm: { json: { compiled_ssm: "compiled-ssm.json", report: "report.json" } },
  posterior: { json: { diagnostics: "diagnostics.json" }, pickle: { fitted: "fitted.pkl" } },
  baseline_ranking: { json: { baseline_ranking: "baseline_ranking.json" } },
  saved_scenarios: { json: { saved_scenarios: "saved_scenarios.json" } },
};

function artifactFileName(artifactId: EpisodeArtifactId, kind: FileKind, key: string): string {
  const filename = ARTIFACT_FILE_SPECS[artifactId][kind]?.[key];
  if (!filename) {
    throw new ArtifactNotFoundError(`${artifactId} has no declared ${kind} file '${key}'`);
  }
  return filename;
}

async function readEpisodeState(workspaceId: string): Promise<EpisodeState> {
  return JSON.parse(await readData(`${workspaceId}/episode/state.json`)) as EpisodeState;
}

async function currentArtifactVersion(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
): Promise<number> {
  const state = await readEpisodeState(workspaceId);
  const info = state.current?.[artifactId];
  if (!info) {
    throw new ArtifactNotFoundError(`No current '${artifactId}' artifact for ${workspaceId}`);
  }
  return info.version;
}

function artifactPath(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
  version: number,
  filename: string,
): string {
  return `${workspaceId}/store/${artifactId}/v${version}/${filename}`;
}

export async function readArtifactJson<T>(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
  key: string,
): Promise<T> {
  const version = await currentArtifactVersion(workspaceId, artifactId);
  const filename = artifactFileName(artifactId, "json", key);
  return JSON.parse(await readData(artifactPath(workspaceId, artifactId, version, filename))) as T;
}

export async function readArtifactBinary(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
  kind: Exclude<FileKind, "json">,
  key: string,
): Promise<{ data: Uint8Array; filename: string }> {
  const version = await currentArtifactVersion(workspaceId, artifactId);
  const filename = artifactFileName(artifactId, kind, key);
  return {
    data: await readBinary(artifactPath(workspaceId, artifactId, version, filename)),
    filename,
  };
}
