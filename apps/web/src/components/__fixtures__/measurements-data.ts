import type { ObservationRecord, MeasurementsData } from "@nof1-causal-lab/api-types";
import measurementsFixture from "./demo-run/measurements.json";
import extractionsSample from "./extraction-sample.json";

// The persisted `measurements.json` no longer inlines observation rows — production derives
// `combined_extractions_sample` and `per_indicator_counts` from the extraction parquet (see
// `deriveMeasurementsData`). `extraction-sample.json` is a parquet-derived sample so stories
// can render the extraction / model-spec panels without reading parquet at module load.
export const combinedExtractionsSample =
  extractionsSample.combined_extractions_sample as unknown as ObservationRecord[];

export const perIndicatorCounts = extractionsSample.per_indicator_counts as Record<string, number>;

export const measurementsData = {
  ...(measurementsFixture as object),
  combined_extractions_sample: combinedExtractionsSample,
  per_indicator_counts: perIndicatorCounts,
} as MeasurementsData;
