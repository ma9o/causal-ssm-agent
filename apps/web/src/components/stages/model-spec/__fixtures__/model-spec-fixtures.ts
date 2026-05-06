import type {
  IndicatorAudit,
  ObservationRecord,
  Stage1bData,
  Stage2Data,
  Stage3Data,
  Stage4Data,
} from "@causal-ssm/api-types";
import { collectStage4UiPriors } from "@/lib/stage4-data";
import { buildStage4LikelihoodDiagnostics } from "@/lib/stage4-likelihood-diagnostics";
import stage1bFixture from "../../../../../../../data/DEMO_HEALTH/run/stage-1b.json";
import stage2Fixture from "../../../../../../../data/DEMO_HEALTH/run/stage-2.json";
import stage3Fixture from "../../../../../../../data/DEMO_HEALTH/run/stage-3.json";
import stage4Fixture from "../../../../../../../data/DEMO_HEALTH/run/stage-4.json";

type DemoHealthStage3Health = {
  indicator: string;
  variance: number | null;
  time_coverage_ratio: number | null;
  max_gap_ratio: number | null;
  dtype_violations: number;
  duplicate_pct: number | null;
  arithmetic_sequence_detected: boolean;
  cell_statuses: Record<string, "ok" | "warning" | "error">;
};

type DemoHealthStage3Fixture = {
  validation_report?: {
    issues?: Stage3Data["dataset_issues"];
    per_indicator_health?: DemoHealthStage3Health[];
  };
};

const stage2 = stage2Fixture as unknown as Stage2Data;
const stage3 = stage3Fixture as unknown as Stage3Data & DemoHealthStage3Fixture;
const stage1b = stage1bFixture as unknown as Stage1bData;
const stage4 = stage4Fixture as unknown as Stage4Data;

function toNumericValue(value: ObservationRecord["value"]): number | null {
  if (value == null) return null;
  if (typeof value === "boolean") return value ? 1 : 0;
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function quantile(sortedValues: number[], p: number): number | null {
  if (sortedValues.length === 0) return null;
  const index = (sortedValues.length - 1) * p;
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  return sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * (index - lower);
}

function buildFixtureIndicatorAudits(): Record<string, IndicatorAudit> {
  const observationsByIndicator = new Map<string, number[]>();
  for (const observation of stage2.combined_extractions_sample) {
    const numericValue = toNumericValue(observation.value);
    if (numericValue == null) continue;
    const existing = observationsByIndicator.get(observation.indicator);
    if (existing) {
      existing.push(numericValue);
    } else {
      observationsByIndicator.set(observation.indicator, [numericValue]);
    }
  }

  const healthByIndicator = new Map(
    (stage3.validation_report?.per_indicator_health ?? []).map((health) => [
      health.indicator,
      health,
    ]),
  );
  const issues = stage3.validation_report?.issues ?? [];
  const dtypeByIndicator = new Map(
    stage1b.causal_spec.measurement.indicators.map((indicator) => [
      indicator.name,
      indicator.measurement_dtype ?? null,
    ]),
  );

  return Object.fromEntries(
    stage4.model_spec.likelihoods.map((likelihood) => {
      const values = [...(observationsByIndicator.get(likelihood.variable) ?? [])].sort(
        (left, right) => left - right,
      );
      const nObs = values.length;
      const mean = nObs > 0 ? values.reduce((sum, value) => sum + value, 0) / nObs : null;
      const variance =
        mean != null && nObs > 1
          ? values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (nObs - 1)
          : nObs === 1
            ? 0
            : null;
      const health = healthByIndicator.get(likelihood.variable);

      return [
        likelihood.variable,
        {
          profile: {
            measurement_dtype: dtypeByIndicator.get(likelihood.variable) ?? null,
            n_obs: nObs,
            mean,
            std: variance == null ? null : Math.sqrt(variance),
            min: values[0] ?? null,
            max: values[values.length - 1] ?? null,
            q25: quantile(values, 0.25),
            q50: quantile(values, 0.5),
            q75: quantile(values, 0.75),
            variance,
            time_coverage_ratio: health?.time_coverage_ratio ?? null,
            max_gap_ratio: health?.max_gap_ratio ?? null,
            dtype_violations: health?.dtype_violations ?? null,
            duplicate_pct: health?.duplicate_pct ?? null,
            arithmetic_sequence_detected: health?.arithmetic_sequence_detected ?? false,
            n_unparseable_timestamps: null,
            zero_fraction:
              nObs > 0 ? values.filter((value) => value === 0).length / nObs : null,
            is_nonnegative: nObs > 0 ? values.every((value) => value >= 0) : null,
            is_unit_interval:
              nObs > 0 ? values.every((value) => value >= 0 && value <= 1) : null,
            looks_integer_valued: nObs > 0 ? values.every(Number.isInteger) : null,
            variance_to_mean_ratio:
              mean != null && mean !== 0 && variance != null ? variance / mean : null,
          },
          validation: {
            issues: issues.filter((issue) => issue.indicator === likelihood.variable),
            checks: health?.cell_statuses ?? {},
          },
        },
      ] satisfies [string, IndicatorAudit];
    }),
  );
}

export const stage4Data = {
  ...(stage4Fixture as object),
  likelihood_diagnostics: buildStage4LikelihoodDiagnostics({
    likelihoods: stage4.model_spec.likelihoods,
    indicatorAudits: buildFixtureIndicatorAudits(),
    observations: stage2.combined_extractions_sample,
  }),
} as Stage4Data;

export const likelihoods = stage4Data.model_spec.likelihoods;
export const parameters = stage4Data.model_spec.parameters;
export const priors = collectStage4UiPriors(stage4Data);
export const indicators = stage1b.causal_spec.measurement.indicators;
export const likelihoodDiagnostics = stage4Data.likelihood_diagnostics;
export const priorPredictiveSamples = stage4Data.prior_predictive_samples as
  | Record<string, number[]>
  | undefined;
