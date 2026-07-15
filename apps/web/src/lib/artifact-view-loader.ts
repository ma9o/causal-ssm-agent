import type {
  CausalDesign,
  RawDataPersistedData,
  MeasurementStructureData,
  MeasurementStructureViewData,
  MeasurementsPersistedData,
  ValidationReportData,
  StatisticalModelSpecPersistedViewData,
  StatisticalModelSpecData,
  PosteriorData,
  ArtifactViewId,
  ArtifactViewData,
  ArtifactViewDataMap,
  BaselineReportData,
  LatentStructureData,
  RawDataData,
  MeasurementsData,
} from "@nof1-causal-lab/api-types";
import { ARTIFACT_VIEW_IDS } from "@nof1-causal-lab/api-types";
import {
  ArtifactNotFoundError,
  readArtifactBinary,
  readArtifactJson,
} from "@/lib/server/artifacts";
import { deriveRawDataData } from "@/lib/raw-data";
import { deriveMeasurementsData } from "@/lib/measurements-data";
import { deriveStatisticalModelSpecData } from "@/lib/model-spec-derived-data";

type ArtifactViewLoader<K extends ArtifactViewId> = (
  workspaceId: string,
) => Promise<ArtifactViewData<K>>;
type ArtifactViewLoaders = { [K in ArtifactViewId]: ArtifactViewLoader<K> };
type CausalDesignArtifactData = { causal_design: CausalDesign };

const ARTIFACT_VIEW_LOADERS = {
  raw_data: async (workspaceId): Promise<RawDataData> => {
    const [payload, raw] = await Promise.all([
      readArtifactJson<RawDataPersistedData>(workspaceId, "raw_data", "profile"),
      readArtifactBinary(workspaceId, "raw_data", "parquet", "raw"),
    ]);
    return deriveRawDataData(payload, raw.data);
  },
  latent_structure: (workspaceId): Promise<LatentStructureData> =>
    readArtifactJson<LatentStructureData>(workspaceId, "latent_structure", "latent_structure"),
  measurement_structure: async (workspaceId): Promise<MeasurementStructureViewData> => {
    const [payload, causalDesignPayload] = await Promise.all([
      readArtifactJson<MeasurementStructureData>(
        workspaceId,
        "measurement_structure",
        "measurement_structure",
      ),
      readArtifactJson<CausalDesignArtifactData>(workspaceId, "causal_design", "causal_design"),
    ]);
    return { ...payload, causal_design: causalDesignPayload.causal_design };
  },
  measurements: async (workspaceId): Promise<MeasurementsData> => {
    const [payload, panel] = await Promise.all([
      readArtifactJson<MeasurementsPersistedData>(workspaceId, "measurements", "measurements"),
      readArtifactBinary(workspaceId, "panel", "parquet", "panel"),
    ]);
    return deriveMeasurementsData(payload, panel.data);
  },
  validation_report: (workspaceId): Promise<ValidationReportData> =>
    readArtifactJson<ValidationReportData>(workspaceId, "validation_report", "validation_report"),
  statistical_model_spec: async (workspaceId): Promise<StatisticalModelSpecData> => {
    const [payload, validationReport, panel] = await Promise.all([
      readArtifactJson<StatisticalModelSpecPersistedViewData>(
        workspaceId,
        "statistical_model_spec",
        "statistical_model_spec",
      ),
      readArtifactJson<ValidationReportData>(workspaceId, "validation_report", "validation_report"),
      readArtifactBinary(workspaceId, "panel", "parquet", "panel"),
    ]);
    return deriveStatisticalModelSpecData(payload, validationReport, panel.data);
  },
  posterior: (workspaceId): Promise<PosteriorData> =>
    readArtifactJson<PosteriorData>(workspaceId, "posterior", "diagnostics"),
  baseline_report: async (workspaceId): Promise<BaselineReportData> => {
    const ranking = await readArtifactJson<BaselineReportData>(
      workspaceId,
      "baseline_report",
      "baseline_report",
    );
    try {
      const saved = await readArtifactJson<{
        scenarios: NonNullable<BaselineReportData["saved_scenarios"]>;
      }>(workspaceId, "saved_scenarios", "saved_scenarios");
      return { ...ranking, saved_scenarios: saved.scenarios };
    } catch (error) {
      if (error instanceof ArtifactNotFoundError) {
        return ranking;
      }
      throw error;
    }
  },
} satisfies ArtifactViewLoaders;

function isArtifactViewId(value: string): value is ArtifactViewId {
  return ARTIFACT_VIEW_IDS.includes(value as ArtifactViewId);
}

export async function loadArtifactView<K extends ArtifactViewId>(
  artifactId: K,
  workspaceId: string,
): Promise<ArtifactViewDataMap[K]>;
export async function loadArtifactView(
  artifactId: string,
  workspaceId: string,
): Promise<ArtifactViewData>;
export async function loadArtifactView(
  artifactId: string,
  workspaceId: string,
): Promise<ArtifactViewData> {
  const loader = isArtifactViewId(artifactId) ? ARTIFACT_VIEW_LOADERS[artifactId] : undefined;
  if (!loader) {
    throw new Error(`Unknown artifact view '${artifactId}'`);
  }
  return loader(workspaceId);
}
