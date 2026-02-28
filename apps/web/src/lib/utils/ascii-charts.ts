import { plot as asciichartPlot } from "asciichart";

interface HistogramOpts {
  width?: number;
  nBins?: number;
  label?: string;
}

/** Render a horizontal bar histogram of numeric values using block characters. */
export function asciiHistogram(values: number[], opts: HistogramOpts = {}): string {
  if (values.length === 0) return "(no data)";

  const { width = 40, nBins = 15, label } = opts;
  const sorted = [...values].sort((a, b) => a - b);
  const min = sorted[0];
  const max = sorted[sorted.length - 1];

  if (min === max) return `All values = ${min.toFixed(3)}`;

  const binWidth = (max - min) / nBins;
  const counts = new Array<number>(nBins).fill(0);
  for (const v of sorted) {
    let idx = Math.floor((v - min) / binWidth);
    if (idx >= nBins) idx = nBins - 1;
    counts[idx]++;
  }

  const maxCount = Math.max(...counts);
  const barScale = maxCount > 0 ? width / maxCount : 0;

  const lines: string[] = [];
  if (label) lines.push(label);

  for (let i = 0; i < nBins; i++) {
    const lo = min + i * binWidth;
    const hi = lo + binWidth;
    const center = (lo + hi) / 2;
    const bar = "\u2588".repeat(Math.round(counts[i] * barScale));
    lines.push(`${center.toFixed(2).padStart(8)} | ${bar} ${counts[i]}`);
  }

  return lines.join("\n");
}

interface DensityOpts {
  height?: number;
  width?: number;
  label?: string;
}

/** Render a density curve using asciichart. */
export function asciiDensity(x: number[], y: number[], opts: DensityOpts = {}): string {
  if (y.length === 0) return "(no data)";

  const { height = 12, width = 60, label } = opts;

  // Resample density to fit desired width
  const resampled = resample(y, width);
  const chart = asciichartPlot(resampled, { height });

  const lines: string[] = [];
  if (label) lines.push(label);
  lines.push(chart);

  // Add x-axis range
  const xMin = x[0];
  const xMax = x[x.length - 1];
  lines.push(`  x: [${xMin.toFixed(3)}, ${xMax.toFixed(3)}]`);

  return lines.join("\n");
}

interface ScatterOpts {
  width?: number;
  height?: number;
  label?: string;
}

/** Render a 2D scatter plot on a character grid. */
export function asciiScatter(
  points: { x: number; y: number; label?: string }[],
  opts: ScatterOpts = {},
): string {
  if (points.length === 0) return "(no data)";

  const { width = 50, height = 20, label } = opts;

  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);

  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;

  // Initialize grid
  const grid: string[][] = Array.from({ length: height }, () => new Array<string>(width).fill(" "));

  // Plot points
  for (const p of points) {
    const col = Math.round(((p.x - xMin) / xRange) * (width - 1));
    const row = height - 1 - Math.round(((p.y - yMin) / yRange) * (height - 1));
    grid[row][col] = "\u2022";
  }

  const lines: string[] = [];
  if (label) lines.push(label);

  // Y-axis label + grid
  for (let r = 0; r < height; r++) {
    const yVal = yMax - (r / (height - 1)) * yRange;
    const prefix = r === 0 || r === height - 1 ? yVal.toFixed(2).padStart(7) : "       ";
    lines.push(`${prefix} |${grid[r].join("")}`);
  }

  // X-axis
  lines.push(`${"       "} +${"\u2500".repeat(width)}`);
  lines.push(
    `${" ".repeat(8)}${xMin.toFixed(2)}${" ".repeat(Math.max(0, width - xMin.toFixed(2).length - xMax.toFixed(2).length))}${xMax.toFixed(2)}`,
  );

  return lines.join("\n");
}

interface MultiLineOpts {
  height?: number;
  width?: number;
  label?: string;
}

/** Render multi-series line chart (e.g. trace plots with multiple chains). */
export function asciiMultiLine(series: number[][], opts: MultiLineOpts = {}): string {
  if (series.length === 0 || series.every((s) => s.length === 0)) return "(no data)";

  const { height = 12, width = 60, label } = opts;

  // Resample each series to fit width
  const resampled = series.map((s) => resample(s, width));
  const chart = asciichartPlot(resampled, { height });

  const lines: string[] = [];
  if (label) lines.push(label);
  lines.push(chart);

  return lines.join("\n");
}

/** Linearly resample an array to a target length. */
function resample(arr: number[], targetLen: number): number[] {
  if (arr.length <= targetLen) return arr;

  const result: number[] = [];
  for (let i = 0; i < targetLen; i++) {
    const srcIdx = (i / (targetLen - 1)) * (arr.length - 1);
    const lo = Math.floor(srcIdx);
    const hi = Math.ceil(srcIdx);
    if (lo === hi) {
      result.push(arr[lo]);
    } else {
      const frac = srcIdx - lo;
      result.push(arr[lo] * (1 - frac) + arr[hi] * frac);
    }
  }
  return result;
}
