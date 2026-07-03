import type { Stage3Data } from "@nof1-causal-lab/api-types";

export function normalizeStage3Data(value: unknown): Stage3Data {
  const stage3 = value as {
    is_valid?: boolean;
    indicators?: Stage3Data["indicators"];
    dataset_issues?: Stage3Data["dataset_issues"];
    validation_report?: {
      is_valid?: boolean;
      issues?: Array<{
        indicator?: string;
        issue_type: string;
        severity: "error" | "warning" | "info";
        message: string;
      }>;
      per_indicator_health?: Array<{
        indicator: string;
        n_obs: number;
        variance: number | null;
        time_coverage_ratio: number | null;
        max_gap_ratio: number | null;
        dtype_violations: number;
        duplicate_pct: number;
        arithmetic_sequence_detected: boolean;
        cell_statuses: Record<string, "ok" | "warning" | "error">;
      }>;
    };
  };

  if (stage3.indicators) return stage3 as Stage3Data;

  const issues = stage3.validation_report?.issues ?? [];
  const profiles = stage3.validation_report?.per_indicator_health ?? [];

  return {
    is_valid: stage3.validation_report?.is_valid ?? stage3.is_valid ?? true,
    dataset_issues: stage3.dataset_issues ?? [],
    indicators: Object.fromEntries(
      profiles.map((profile) => [
        profile.indicator,
        {
          profile: {
            measurement_dtype: null,
            n_obs: profile.n_obs,
            mean: null,
            std: null,
            min: null,
            max: null,
            q25: null,
            q50: null,
            q75: null,
            variance: profile.variance,
            time_coverage_ratio: profile.time_coverage_ratio,
            max_gap_ratio: profile.max_gap_ratio,
            dtype_violations: profile.dtype_violations,
            duplicate_pct: profile.duplicate_pct,
            arithmetic_sequence_detected: profile.arithmetic_sequence_detected,
            n_unparseable_timestamps: null,
            zero_fraction: null,
            is_nonnegative: null,
            is_unit_interval: null,
            looks_integer_valued: null,
            variance_to_mean_ratio: null,
          },
          validation: {
            issues: issues.filter((issue) => issue.indicator === profile.indicator),
            checks: profile.cell_statuses,
          },
        },
      ]),
    ),
  };
}
