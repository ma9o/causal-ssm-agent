import type {
  CausalDesign,
  Stage0PersistedData,
  Stage1bData,
  Stage1bViewData,
  Stage2PersistedData,
  Stage3Data,
  Stage4PersistedData,
  StageId,
  Stage6Data,
} from "@nof1-causal-lab/api-types";
import {
  ArtifactNotFoundError,
  readArtifactBinary,
  readArtifactJson,
} from "@/lib/server/artifacts";
import { deriveStage0Data } from "@/lib/stage0-data";
import { deriveStage2Data } from "@/lib/stage2-data";
import { deriveStage4Data } from "@/lib/stage4-derived-data";

type StageResultLoader = (workspaceId: string) => Promise<unknown>;

const STAGE_RESULT_LOADERS: Partial<Record<StageId, StageResultLoader>> = {
  "stage-0": async (workspaceId) => {
    const [payload, raw] = await Promise.all([
      readArtifactJson<Stage0PersistedData>(workspaceId, "raw_data", "profile"),
      readArtifactBinary(workspaceId, "raw_data", "parquet", "raw"),
    ]);
    return deriveStage0Data(payload, raw.data);
  },
  "stage-1a": (workspaceId) =>
    readArtifactJson(workspaceId, "latent_structure", "latent_structure"),
  "stage-1b": async (workspaceId): Promise<Stage1bViewData> => {
    const [payload, causalDesign] = await Promise.all([
      readArtifactJson<Stage1bData>(workspaceId, "measurement_structure", "measurement_structure"),
      readArtifactJson<CausalDesign>(workspaceId, "causal_design", "causal_design"),
    ]);
    return { ...payload, causal_design: causalDesign };
  },
  "stage-2": async (workspaceId) => {
    const [payload, panel] = await Promise.all([
      readArtifactJson<Stage2PersistedData>(workspaceId, "measurements", "measurements"),
      readArtifactBinary(workspaceId, "panel", "parquet", "panel"),
    ]);
    return deriveStage2Data(payload, panel.data);
  },
  "stage-3": (workspaceId) =>
    readArtifactJson(workspaceId, "validation_report", "validation_report"),
  "stage-4": async (workspaceId) => {
    const [payload, stage3, panel] = await Promise.all([
      readArtifactJson<Stage4PersistedData>(
        workspaceId,
        "statistical_model_spec",
        "statistical_model_spec",
      ),
      readArtifactJson<Stage3Data>(workspaceId, "validation_report", "validation_report"),
      readArtifactBinary(workspaceId, "panel", "parquet", "panel"),
    ]);
    return deriveStage4Data(payload, stage3, panel.data);
  },
  "stage-5b": (workspaceId) => readArtifactJson(workspaceId, "posterior", "diagnostics"),
  "stage-6": async (workspaceId) => {
    const ranking = await readArtifactJson<Stage6Data>(
      workspaceId,
      "baseline_report",
      "baseline_report",
    );
    try {
      const saved = await readArtifactJson<{
        scenarios: NonNullable<Stage6Data["saved_scenarios"]>;
      }>(workspaceId, "saved_scenarios", "saved_scenarios");
      return { ...ranking, saved_scenarios: saved.scenarios };
    } catch (error) {
      if (error instanceof ArtifactNotFoundError) {
        return ranking;
      }
      throw error;
    }
  },
};

export async function loadStageResult(stageId: string, workspaceId: string): Promise<unknown> {
  const loader = STAGE_RESULT_LOADERS[stageId as StageId];
  if (!loader) {
    throw new Error(`Unknown stage '${stageId}'`);
  }
  return loader(workspaceId);
}
