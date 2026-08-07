import type {
  ObservationRecord,
  ValidationReportData,
  StatisticalModelSpecData,
  StatisticalModelSpecPersistedViewData,
} from "@nof1-causal-lab/api-types";
import { buildModelSpecLikelihoodDiagnostics } from "./model-spec-likelihood-diagnostics";
import {
  normalizeParquetScalar,
  type ParquetMetadata,
  readParquetRows,
  toArrayBuffer,
} from "./parquet-utils";

async function readObservationRecords(parquetBytes: Uint8Array): Promise<ObservationRecord[]> {
  const file = toArrayBuffer(parquetBytes);
  const { parquetMetadata } = await import("hyparquet");
  const metadata = parquetMetadata(file) as ParquetMetadata;
  const totalRows = Number(metadata.num_rows);

  if (totalRows === 0) {
    return [];
  }

  const rows = await readParquetRows(file, metadata, 0, totalRows, ["indicator", "value"]);
  return rows.map((row) => ({
    indicator: String(row.indicator ?? ""),
    value: normalizeParquetScalar(row.value),
    anchor_time: null,
  }));
}

export async function deriveStatisticalModelSpecData(
  payload: StatisticalModelSpecPersistedViewData,
  validationReport: ValidationReportData,
  parquetBytes: Uint8Array,
): Promise<StatisticalModelSpecData> {
  const observations = await readObservationRecords(parquetBytes);

  return {
    ...payload,
    likelihood_diagnostics: buildModelSpecLikelihoodDiagnostics({
      likelihoods: payload.statistical_model_spec.likelihoods,
      indicatorAudits: validationReport.indicators,
      observations,
    }),
  };
}
