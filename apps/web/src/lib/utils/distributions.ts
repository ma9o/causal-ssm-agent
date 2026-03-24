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

export function evaluatePdf(
  distribution: string,
  params: DistParams,
  nPoints = 200,
): Array<{ x: number; y: number }> {
  let xMin = -4;
  let xMax = 4;

  // Set range based on distribution
  if (distribution === "Normal") {
    const mu = params.mu ?? 0;
    const sigma = params.sigma ?? 1;
    xMin = mu - 4 * sigma;
    xMax = mu + 4 * sigma;
  } else if (distribution === "HalfNormal") {
    xMin = 0;
    xMax = (params.sigma ?? 1) * 4;
  } else if (distribution === "TruncatedNormal") {
    xMin = params.lower ?? -1;
    xMax = params.upper ?? 1;
  } else if (distribution === "Gamma") {
    xMin = 0;
    xMax = ((params.concentration ?? 2) / (params.rate ?? 1)) * 3;
  } else if (distribution === "LogNormal") {
    xMin = 0;
    xMax = Math.exp((params.mu ?? 0) + 4 * (params.sigma ?? 1));
  } else if (distribution === "Exponential") {
    xMin = 0;
    xMax = 5 / (params.rate ?? 1);
  } else if (distribution === "Beta") {
    xMin = 0.001;
    xMax = 0.999;
  } else if (distribution === "Uniform") {
    xMin = params.lower ?? 0;
    xMax = params.upper ?? 1;
  }

  const step = (xMax - xMin) / nPoints;
  const points: Array<{ x: number; y: number }> = [];

  for (let i = 0; i <= nPoints; i++) {
    const x = xMin + i * step;
    let y = 0;

    if (distribution === "Normal") {
      y = jStat.normal.pdf(
        x,
        params.mu ?? 0,
        params.sigma ?? 1,
      );
    } else if (distribution === "HalfNormal") {
      y = halfNormalPdf(x, params.sigma ?? 1);
    } else if (distribution === "TruncatedNormal") {
      y = truncatedNormalPdf(
        x,
        params.mu ?? 0,
        params.sigma ?? 1,
        params.lower ?? -1,
        params.upper ?? 1,
      );
    } else if (distribution === "Gamma") {
      y = jStat.gamma.pdf(x, params.concentration ?? 2, 1 / (params.rate ?? 1));
    } else if (distribution === "LogNormal") {
      y = logNormalPdf(x, params.mu ?? 0, params.sigma ?? 1);
    } else if (distribution === "Exponential") {
      y = exponentialPdf(x, params.rate ?? 1);
    } else if (distribution === "Beta") {
      y = jStat.beta.pdf(x, params.alpha ?? 2, params.beta ?? 2);
    } else if (distribution === "Uniform") {
      const uLow = params.lower ?? 0;
      const uHigh = params.upper ?? 1;
      const uRange = uHigh - uLow;
      y = uRange > 0 && x >= uLow && x <= uHigh ? 1 / uRange : 0;
    }

    points.push({ x: Math.round(x * 1000) / 1000, y });
  }

  return points;
}
