/** Evaluate PDF for chart rendering — visualization only, not analytical logic. */

import jStat from "jstat";
import type { PriorProposal } from "@nof1-causal-lab/api-types";

type PriorParams = PriorProposal["params"];

function priorParam(params: PriorParams, name: string, defaultValue: number): number {
  const entry = Object.entries(params).find(([key]) => key === name);
  return typeof entry?.[1] === "number" ? entry[1] : defaultValue;
}

function halfNormalPdf(x: number, sigma: number): number {
  if (x < 0) return 0;
  return 2 * jStat.normal.pdf(x, 0, sigma);
}

function logNormalPdf(x: number, mu: number, sigma: number): number {
  if (x <= 0) return 0;
  return (
    Math.exp(-((Math.log(x) - mu) ** 2) / (2 * sigma * sigma)) /
    (x * sigma * Math.sqrt(2 * Math.PI))
  );
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
  params: PriorParams,
  nPoints = 200,
): Array<{ x: number; y: number }> {
  let xMin = -4;
  let xMax = 4;

  // Set range based on distribution
  if (distribution === "Normal") {
    const mu = priorParam(params, "mu", 0);
    const sigma = priorParam(params, "sigma", 1);
    xMin = mu - 4 * sigma;
    xMax = mu + 4 * sigma;
  } else if (distribution === "HalfNormal") {
    xMin = 0;
    xMax = priorParam(params, "sigma", 1) * 4;
  } else if (distribution === "TruncatedNormal") {
    xMin = priorParam(params, "lower", -1);
    xMax = priorParam(params, "upper", 1);
  } else if (distribution === "Gamma") {
    xMin = 0;
    xMax = (priorParam(params, "concentration", 2) / priorParam(params, "rate", 1)) * 3;
  } else if (distribution === "LogNormal") {
    xMin = 0;
    xMax = Math.exp(priorParam(params, "mu", 0) + 4 * priorParam(params, "sigma", 1));
  } else if (distribution === "Exponential") {
    xMin = 0;
    xMax = 5 / priorParam(params, "rate", 1);
  } else if (distribution === "Beta") {
    xMin = 0.001;
    xMax = 0.999;
  } else if (distribution === "Uniform") {
    xMin = priorParam(params, "lower", 0);
    xMax = priorParam(params, "upper", 1);
  }

  const step = (xMax - xMin) / nPoints;
  const points: Array<{ x: number; y: number }> = [];

  for (let i = 0; i <= nPoints; i++) {
    const x = xMin + i * step;
    let y = 0;

    if (distribution === "Normal") {
      y = jStat.normal.pdf(x, priorParam(params, "mu", 0), priorParam(params, "sigma", 1));
    } else if (distribution === "HalfNormal") {
      y = halfNormalPdf(x, priorParam(params, "sigma", 1));
    } else if (distribution === "TruncatedNormal") {
      y = truncatedNormalPdf(
        x,
        priorParam(params, "mu", 0),
        priorParam(params, "sigma", 1),
        priorParam(params, "lower", -1),
        priorParam(params, "upper", 1),
      );
    } else if (distribution === "Gamma") {
      y = jStat.gamma.pdf(
        x,
        priorParam(params, "concentration", 2),
        1 / priorParam(params, "rate", 1),
      );
    } else if (distribution === "LogNormal") {
      y = logNormalPdf(x, priorParam(params, "mu", 0), priorParam(params, "sigma", 1));
    } else if (distribution === "Exponential") {
      y = exponentialPdf(x, priorParam(params, "rate", 1));
    } else if (distribution === "Beta") {
      y = jStat.beta.pdf(x, priorParam(params, "alpha", 2), priorParam(params, "beta", 2));
    } else if (distribution === "Uniform") {
      const uLow = priorParam(params, "lower", 0);
      const uHigh = priorParam(params, "upper", 1);
      const uRange = uHigh - uLow;
      y = uRange > 0 && x >= uLow && x <= uHigh ? 1 / uRange : 0;
    }

    points.push({ x: Math.round(x * 1000) / 1000, y });
  }

  return points;
}
