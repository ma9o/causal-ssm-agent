/**
 * Build a histogram from numeric values.
 *
 * Returns bin center, count, and edges for each bin.
 */
import { bin } from "d3-array";

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
