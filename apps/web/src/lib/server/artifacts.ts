import { getToolServerUrl } from "@/lib/runtime-urls";
import type { ArtifactEnvelope } from "@nof1-causal-lab/api-types";
import type { EpisodeArtifactId } from "@/lib/episode-types";

export type { EpisodeArtifactId } from "@/lib/episode-types";

type FileKind = "json" | "parquet" | "pickle";

type ArtifactFileSpec = {
  json?: Record<string, string>;
  parquet?: Record<string, string>;
  pickle?: Record<string, string>;
};

export class ArtifactNotFoundError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ArtifactNotFoundError";
  }
}

const TOOL_SERVER = getToolServerUrl();

const ARTIFACT_FILE_SPECS: Record<EpisodeArtifactId, ArtifactFileSpec> = {
  question: { json: { question: "question.json" } },
  raw_data: { json: { profile: "profile.json" }, parquet: { raw: "raw.parquet" } },
  latent_structure: { json: { latent_structure: "latent-structure.json" } },
  measurement_structure: { json: { measurement_structure: "measurement_structure.json" } },
  causal_design: { json: { causal_design: "causal_design.json" } },
  identification_report: { json: { identification_report: "identification_report.json" } },
  measurements: { json: { measurements: "measurements.json" } },
  panel: { parquet: { panel: "panel.parquet" } },
  validation_report: { json: { validation_report: "validation_report.json" } },
  statistical_model_spec: { json: { statistical_model_spec: "statistical_model_spec.json" } },
  compiled_ssm: { json: { compiled_ssm: "compiled-ssm.json", report: "report.json" } },
  posterior: { json: { diagnostics: "diagnostics.json" }, pickle: { fitted: "fitted.pkl" } },
  baseline_report: { json: { baseline_report: "baseline_report.json" } },
  saved_scenarios: { json: { saved_scenarios: "saved_scenarios.json" } },
};

function artifactFileName(artifactId: EpisodeArtifactId, kind: FileKind, key: string): string {
  const filename = ARTIFACT_FILE_SPECS[artifactId][kind]?.[key];
  if (!filename) {
    throw new ArtifactNotFoundError(`${artifactId} has no declared ${kind} file '${key}'`);
  }
  return filename;
}

async function fetchArtifact(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
): Promise<ArtifactEnvelope> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${encodeURIComponent(workspaceId)}/artifacts/${encodeURIComponent(
      artifactId,
    )}`,
    { cache: "no-store" },
  );
  if (response.status === 404) {
    throw new ArtifactNotFoundError(await response.text());
  }
  if (!response.ok) {
    throw new Error(`Artifact facade error ${response.status}: ${await response.text()}`);
  }
  return response.json() as Promise<ArtifactEnvelope>;
}

async function fetchArtifactFile(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
  filename: string,
): Promise<Uint8Array> {
  const response = await fetch(
    `${TOOL_SERVER}/api/episodes/${encodeURIComponent(workspaceId)}/artifacts/${encodeURIComponent(
      artifactId,
    )}/files/${encodeURIComponent(filename)}`,
    { cache: "no-store" },
  );
  if (response.status === 404) {
    throw new ArtifactNotFoundError(await response.text());
  }
  if (!response.ok) {
    throw new Error(`Artifact facade file error ${response.status}: ${await response.text()}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}

export async function readArtifactJson<T>(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
  key: string,
): Promise<T> {
  const filename = artifactFileName(artifactId, "json", key);
  const artifact = await fetchArtifact(workspaceId, artifactId);
  if (!(filename in artifact.payload)) {
    throw new ArtifactNotFoundError(`${artifactId} has no payload file '${filename}'`);
  }
  return artifact.payload[filename] as T;
}

export async function readArtifactBinary(
  workspaceId: string,
  artifactId: EpisodeArtifactId,
  kind: Exclude<FileKind, "json">,
  key: string,
): Promise<{ data: Uint8Array; filename: string }> {
  const filename = artifactFileName(artifactId, kind, key);
  return {
    data: await fetchArtifactFile(workspaceId, artifactId, filename),
    filename,
  };
}
