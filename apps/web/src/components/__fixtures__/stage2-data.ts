import type { ObservationRecord, Stage2Data } from "@nof1-causal-lab/api-types";
import stage2Fixture from "./demo-run/stage-2.json";
import extractionsSample from "./stage2-extractions-sample.json";

// The persisted `stage-2.json` no longer inlines observation rows — production derives
// `combined_extractions_sample` and `per_indicator_counts` from the Stage 2 parquet (see
// `deriveStage2Data`). `stage2-extractions-sample.json` is a parquet-derived sample so stories
// can render the Stage 2 / Stage 4 panels without reading parquet at module load.
export const combinedExtractionsSample =
  extractionsSample.combined_extractions_sample as unknown as ObservationRecord[];

export const perIndicatorCounts = extractionsSample.per_indicator_counts as Record<string, number>;

export const stage2Data = {
  ...(stage2Fixture as object),
  combined_extractions_sample: combinedExtractionsSample,
  per_indicator_counts: perIndicatorCounts,
} as Stage2Data;
