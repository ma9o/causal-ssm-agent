import { buildHistogram } from "@/lib/utils/histogram";
import type {
  IndicatorAudit,
  LikelihoodSpec,
  ObservationRecord,
  Stage4LikelihoodDiagnostics,
} from "@causal-ssm/api-types";

function toNumericValue(value: ObservationRecord["value"]): number | null {
  if (value == null) {
    return null;
  }
  if (typeof value === "boolean") {
    return value ? 1 : 0;
  }
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : null;
}

function isDiscreteLikelihood(likelihood: LikelihoodSpec): boolean {
  return (
    likelihood.distribution === "poisson" ||
    likelihood.distribution === "bernoulli" ||
    likelihood.distribution === "negative_binomial" ||
    likelihood.distribution === "ordered_logistic"
  );
}

function buildCountFrequency(values: number[]) {
  const counts = new Map<number, number>();
  for (const value of values) {
    counts.set(value, (counts.get(value) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .sort(([left], [right]) => left - right)
    .map(([binCenter, count]) => ({ binCenter, count }));
}

export function buildStage4LikelihoodDiagnostics({
  likelihoods,
  indicatorAudits,
  observations,
}: {
  likelihoods: LikelihoodSpec[];
  indicatorAudits?: Record<string, IndicatorAudit | undefined>;
  observations: ObservationRecord[];
}): Record<string, Stage4LikelihoodDiagnostics> {
  const observationsByIndicator = new Map<string, number[]>();

  for (const observation of observations) {
    const numericValue = toNumericValue(observation.value);
    if (numericValue == null) {
      continue;
    }
    const existing = observationsByIndicator.get(observation.indicator);
    if (existing) {
      existing.push(numericValue);
      continue;
    }
    observationsByIndicator.set(observation.indicator, [numericValue]);
  }

  return Object.fromEntries(
    likelihoods.map((likelihood) => {
      const numericValues = observationsByIndicator.get(likelihood.variable) ?? [];
      const histogram = isDiscreteLikelihood(likelihood)
        ? buildCountFrequency(numericValues)
        : buildHistogram(numericValues, Math.min(15, Math.ceil(Math.sqrt(numericValues.length)))).map(
            ({ binCenter, count }) => ({ binCenter, count }),
          );

      return [
        likelihood.variable,
        {
          variable: likelihood.variable,
          profile: indicatorAudits?.[likelihood.variable]?.profile ?? null,
          histogram,
        },
      ] satisfies [string, Stage4LikelihoodDiagnostics];
    }),
  );
}
