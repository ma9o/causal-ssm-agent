const NO_DATA = "(no data)";

// ── Shared helpers ──────────────────────────────────────────────────────

function fmt(v: number, decimals = 3): string {
  return v.toFixed(decimals);
}

function summaryStats(values: number[]): { mean: number; sd: number; min: number; max: number; median: number } {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const sd = Math.sqrt(variance);
  const median = n % 2 === 1 ? sorted[Math.floor(n / 2)] : (sorted[n / 2 - 1] + sorted[n / 2]) / 2;
  return { mean, sd, min: sorted[0], max: sorted[n - 1], median };
}

function statsLine(values: number[]): string {
  const s = summaryStats(values);
  return `  n=${values.length}  mean=${fmt(s.mean)}  sd=${fmt(s.sd)}  median=${fmt(s.median)}  range=[${fmt(s.min)}, ${fmt(s.max)}]`;
}

/** Linearly resample an array to a target length. */
function resample(arr: number[], targetLen: number): number[] {
  if (arr.length <= targetLen) return arr;
  if (targetLen <= 0) return [];
  if (targetLen === 1) return [arr[0]];

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

// ── Line chart (replaces asciichart dependency) ─────────────────────────

/** Characters for line drawing, indexed by (enters-from-below, exits-above). */
const LINE_CHARS: Record<string, string> = {
  rise: "╱",
  fall: "╲",
  flat: "─",
  up_turn: "╰",
  down_turn: "╮",
  vert: "│",
};

/**
 * Render a single-series line chart on a character grid.
 * Inspired by asciichart but dependency-free.
 */
function plotLine(
  series: number[],
  opts: { height: number; yMin: number; yMax: number; width: number },
): string[][] {
  const { height, yMin, yMax, width } = opts;
  const yRange = yMax - yMin || 1;

  // Initialize grid
  const grid: string[][] = Array.from({ length: height }, () => new Array<string>(width).fill(" "));

  const toRow = (v: number): number => {
    const row = height - 1 - Math.round(((v - yMin) / yRange) * (height - 1));
    return Math.max(0, Math.min(height - 1, row));
  };

  for (let col = 0; col < series.length; col++) {
    const row = toRow(series[col]);
    grid[row][col] = "•";

    // Connect to previous point
    if (col > 0) {
      const prevRow = toRow(series[col - 1]);
      if (prevRow === row) {
        // Horizontal — the dot already covers it
      } else {
        // Vertical fill between points
        const rMin = Math.min(prevRow, row);
        const rMax = Math.max(prevRow, row);
        for (let r = rMin + 1; r < rMax; r++) {
          if (grid[r][col] === " ") grid[r][col] = "│";
        }
      }
    }
  }

  return grid;
}

/**
 * Render one or more series as a line chart with Y-axis labels.
 * Multi-series uses different markers to distinguish chains.
 */
function renderLineChart(
  allSeries: number[][],
  opts: { height: number; width: number },
): string[] {
  const { height, width } = opts;

  // Global y bounds across all series
  const allVals = allSeries.flat();
  if (allVals.length === 0) return [];
  const yMin = Math.min(...allVals);
  const yMax = Math.max(...allVals);

  // Markers for multi-series
  const markers = ["•", "◦", "×", "+", "◆", "■", "▲", "▼"];

  // Initialize composite grid
  const grid: string[][] = Array.from({ length: height }, () => new Array<string>(width).fill(" "));
  const yRange = yMax - yMin || 1;

  const toRow = (v: number): number => {
    const row = height - 1 - Math.round(((v - yMin) / yRange) * (height - 1));
    return Math.max(0, Math.min(height - 1, row));
  };

  for (let si = 0; si < allSeries.length; si++) {
    const series = allSeries[si];
    const marker = markers[si % markers.length];

    for (let col = 0; col < series.length; col++) {
      const row = toRow(series[col]);
      grid[row][col] = marker;

      if (col > 0) {
        const prevRow = toRow(series[col - 1]);
        const rMin = Math.min(prevRow, row);
        const rMax = Math.max(prevRow, row);
        for (let r = rMin + 1; r < rMax; r++) {
          if (grid[r][col] === " ") grid[r][col] = "│";
        }
      }
    }
  }

  // Render with Y-axis
  const lines: string[] = [];
  const labelWidth = 8;
  for (let r = 0; r < height; r++) {
    const yVal = yMax - (r / Math.max(height - 1, 1)) * yRange;
    const prefix = r === 0 || r === height - 1 ? fmt(yVal, 2).padStart(labelWidth) : " ".repeat(labelWidth);
    lines.push(`${prefix} │${grid[r].join("")}`);
  }

  // X-axis
  lines.push(`${" ".repeat(labelWidth)} └${"─".repeat(width)}`);

  return lines;
}

// ── Public chart functions ──────────────────────────────────────────────

interface HistogramOpts {
  width?: number;
  nBins?: number;
  label?: string;
}

/** Render a horizontal bar histogram of numeric values using block characters. */
export function asciiHistogram(values: number[], opts: HistogramOpts = {}): string {
  const finite = values.filter((v) => v != null && Number.isFinite(v));
  if (finite.length === 0) return NO_DATA;

  const { width = 40, nBins = 15, label } = opts;
  const sorted = [...finite].sort((a, b) => a - b);
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
    const bar = "█".repeat(Math.round(counts[i] * barScale));
    lines.push(`${center.toFixed(2).padStart(8)} │ ${bar} ${counts[i]}`);
  }

  lines.push(statsLine(values));

  return lines.join("\n");
}

interface DensityOpts {
  height?: number;
  width?: number;
  label?: string;
}

/** Render a density curve as a line chart with summary stats. */
export function asciiDensity(x: number[], y: number[], opts: DensityOpts = {}): string {
  if (y.length === 0) return NO_DATA;

  const { height = 12, width = 60, label } = opts;

  const resampled = resample(y, width);
  const chartLines = renderLineChart([resampled], { height, width });

  const lines: string[] = [];
  if (label) lines.push(label);
  lines.push(...chartLines);

  // X-axis range
  const xMin = x[0];
  const xMax = x[x.length - 1];
  lines.push(`  x: [${fmt(xMin)}, ${fmt(xMax)}]`);

  // Density-weighted stats (treat as distribution)
  const totalArea = y.reduce((a, b) => a + b, 0);
  if (totalArea > 0 && x.length === y.length) {
    const wMean = x.reduce((acc, xi, i) => acc + xi * y[i], 0) / totalArea;
    const wVar = x.reduce((acc, xi, i) => acc + (xi - wMean) ** 2 * y[i], 0) / totalArea;
    const peakIdx = y.indexOf(Math.max(...y));
    lines.push(`  mean=${fmt(wMean)}  sd=${fmt(Math.sqrt(wVar))}  mode=${fmt(x[peakIdx])}`);
  }

  return lines.join("\n");
}

interface ScatterOpts {
  width?: number;
  height?: number;
  label?: string;
}

/** Render a 2D scatter plot on a character grid with summary stats. */
export function asciiScatter(
  points: { x: number; y: number; label?: string }[],
  opts: ScatterOpts = {},
): string {
  if (points.length === 0) return NO_DATA;

  const { width = 50, height: rawHeight = 20, label } = opts;
  const height = Math.max(2, rawHeight);

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
    grid[row][col] = "•";
  }

  const lines: string[] = [];
  if (label) lines.push(label);

  // Y-axis label + grid
  for (let r = 0; r < height; r++) {
    const yVal = yMax - (r / (height - 1)) * yRange;
    const prefix = r === 0 || r === height - 1 ? yVal.toFixed(2).padStart(7) : "       ";
    lines.push(`${prefix} │${grid[r].join("")}`);
  }

  // X-axis
  lines.push(`${"       "} └${"─".repeat(width)}`);
  lines.push(
    `${" ".repeat(8)}${xMin.toFixed(2)}${" ".repeat(Math.max(0, width - xMin.toFixed(2).length - xMax.toFixed(2).length))}${xMax.toFixed(2)}`,
  );

  // Summary stats for both axes
  const xStats = summaryStats(xs);
  const yStats = summaryStats(ys);
  lines.push(`  n=${points.length}  x: mean=${fmt(xStats.mean)} sd=${fmt(xStats.sd)}  y: mean=${fmt(yStats.mean)} sd=${fmt(yStats.sd)}`);

  return lines.join("\n");
}

interface MultiLineOpts {
  height?: number;
  width?: number;
  label?: string;
}

/** Render multi-series line chart (e.g. trace plots with multiple chains) with per-series stats. */
export function asciiMultiLine(series: number[][], opts: MultiLineOpts = {}): string {
  if (series.length === 0 || series.every((s) => s.length === 0)) return NO_DATA;

  const { height = 12, width = 60, label } = opts;

  // Resample each series to fit width
  const resampled = series.map((s) => resample(s, width));
  const chartLines = renderLineChart(resampled, { height, width });

  const lines: string[] = [];
  if (label) lines.push(label);
  lines.push(...chartLines);

  // Per-series summary stats
  const markers = ["•", "◦", "×", "+", "◆", "■", "▲", "▼"];
  for (let i = 0; i < series.length; i++) {
    const s = series[i];
    if (s.length === 0) continue;
    const stats = summaryStats(s);
    const marker = markers[i % markers.length];
    lines.push(`  ${marker} series ${i + 1}: n=${s.length}  mean=${fmt(stats.mean)}  sd=${fmt(stats.sd)}  range=[${fmt(stats.min)}, ${fmt(stats.max)}]`);
  }

  return lines.join("\n");
}
