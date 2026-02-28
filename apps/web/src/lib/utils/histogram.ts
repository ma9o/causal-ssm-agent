/**
 * Numeric statistical utilities: histograms, quantiles, etc.
 */
import { bin } from "d3-array";

/** Linear-interpolation quantile on a pre-sorted array. */
export function quantile(sorted: number[], q: number): number {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

export function buildHistogram(
  values: number[],
  nBins = 20,
): Array<{ binCenter: number; count: number; binStart: number; binEnd: number }> {
  if (values.length === 0) return [];

  const histogram = bin().thresholds(nBins);
  const bins = histogram(values);

  return bins.map((b) => {
    const binStart = b.x0 ?? 0;
    const binEnd = b.x1 ?? 0;
    return {
      binCenter: Math.round(((binStart + binEnd) / 2) * 100) / 100,
      count: b.length,
      binStart,
      binEnd,
    };
  });
}
