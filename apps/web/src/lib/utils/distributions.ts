/** Evaluate PDF for chart rendering — visualization only, not analytical logic. */

import jStat from "jstat";

type DistParams = Record<string, number>;

function halfNormalPdf(x: number, sigma: number): number {
  if (x < 0) return 0;
  return 2 * jStat.normal.pdf(x, 0, sigma);
}

function logNormalPdf(x: number, mu: number, sigma: number): number {
  if (x <= 0) return 0;
  return Math.exp(-((Math.log(x) - mu) ** 2) / (2 * sigma * sigma)) / (x * sigma * Math.sqrt(2 * Math.PI));
}

function exponentialPdf(x: number, rate: number): number {
  if (x < 0) return 0;
  return rate * Math.exp(-rate * x);
}

function truncatedNormalPdf(
  x: number,
  mu: number,
  sigma: number,
  lower: number,
  upper: number,
): number {
  if (x < lower || x > upper || lower >= upper) return 0;
  const zLower = (lower - mu) / sigma;
  const zUpper = (upper - mu) / sigma;
  const normalizer = jStat.normal.cdf(zUpper, 0, 1) - jStat.normal.cdf(zLower, 0, 1);
  if (normalizer <= 0) return 0;
  return jStat.normal.pdf(x, mu, sigma) / normalizer;
}

function normalizeDistributionName(distribution: string): string {
  return distribution.toLowerCase().replace(/[\s-]/g, "_");
}

export function evaluatePdf(
  distribution: string,
  params: DistParams,
  nPoints = 200,
): Array<{ x: number; y: number }> {
  const dist = normalizeDistributionName(distribution);
  let xMin = -4;
  let xMax = 4;

  // Set range based on distribution
  if (dist === "normal" || dist === "gaussian") {
    const mu = params.mu ?? params.loc ?? 0;
    const sigma = params.sigma ?? params.scale ?? 1;
    xMin = mu - 4 * sigma;
    xMax = mu + 4 * sigma;
  } else if (dist === "halfnormal" || dist === "half_normal") {
    xMin = 0;
    xMax = (params.sigma ?? params.scale ?? 1) * 4;
  } else if (dist === "truncatednormal" || dist === "truncated_normal") {
    xMin = params.lower ?? params.low ?? -1;
    xMax = params.upper ?? params.high ?? 1;
  } else if (dist === "gamma") {
    xMin = 0;
    xMax = ((params.alpha ?? params.concentration ?? 2) / (params.beta ?? params.rate ?? 1)) * 3;
  } else if (dist === "lognormal" || dist === "log_normal") {
    xMin = 0;
    xMax = Math.exp((params.mu ?? params.loc ?? 0) + 4 * (params.sigma ?? params.scale ?? 1));
  } else if (dist === "exponential") {
    xMin = 0;
    xMax = 5 / (params.rate ?? 1);
  } else if (dist === "beta") {
    xMin = 0.001;
    xMax = 0.999;
  } else if (dist === "uniform") {
    xMin = params.lower ?? params.low ?? 0;
    xMax = params.upper ?? params.high ?? 1;
  }

  const step = (xMax - xMin) / nPoints;
  const points: Array<{ x: number; y: number }> = [];

  for (let i = 0; i <= nPoints; i++) {
    const x = xMin + i * step;
    let y = 0;

    if (dist === "normal" || dist === "gaussian") {
      y = jStat.normal.pdf(
        x,
        params.mu ?? params.loc ?? 0,
        params.sigma ?? params.scale ?? 1,
      );
    } else if (dist === "halfnormal" || dist === "half_normal") {
      y = halfNormalPdf(x, params.sigma ?? params.scale ?? 1);
    } else if (dist === "truncatednormal" || dist === "truncated_normal") {
      y = truncatedNormalPdf(
        x,
        params.mu ?? params.loc ?? 0,
        params.sigma ?? params.scale ?? 1,
        params.lower ?? params.low ?? -1,
        params.upper ?? params.high ?? 1,
      );
    } else if (dist === "gamma") {
      const alpha = params.alpha ?? params.concentration ?? 2;
      const rate = params.beta ?? params.rate ?? 1;
      y = jStat.gamma.pdf(x, alpha, 1 / rate);
    } else if (dist === "lognormal" || dist === "log_normal") {
      y = logNormalPdf(x, params.mu ?? params.loc ?? 0, params.sigma ?? params.scale ?? 1);
    } else if (dist === "exponential") {
      y = exponentialPdf(x, params.rate ?? 1);
    } else if (dist === "beta") {
      y = jStat.beta.pdf(x, params.alpha ?? params.a ?? 2, params.beta ?? params.b ?? 2);
    } else if (dist === "uniform") {
      const uLow = params.lower ?? params.low ?? 0;
      const uHigh = params.upper ?? params.high ?? 1;
      const uRange = uHigh - uLow;
      y = uRange > 0 && x >= uLow && x <= uHigh ? 1 / uRange : 0;
    }

    points.push({ x: Math.round(x * 1000) / 1000, y });
  }

  return points;
}
