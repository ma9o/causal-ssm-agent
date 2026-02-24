/** Evaluate PDF for chart rendering — visualization only, not analytical logic. */

// @ts-expect-error -- jstat has no bundled types
import jStat from "jstat";

type DistParams = Record<string, number>;

function halfNormalPdf(x: number, sigma: number): number {
  if (x < 0) return 0;
  return 2 * (jStat.normal.pdf(x, 0, sigma) as number);
}

export function evaluatePdf(
  distribution: string,
  params: DistParams,
  nPoints = 200,
): Array<{ x: number; y: number }> {
  const dist = distribution.toLowerCase();
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
  } else if (dist === "gamma") {
    xMin = 0;
    xMax = ((params.alpha ?? params.concentration ?? 2) / (params.beta ?? params.rate ?? 1)) * 3;
  } else if (dist === "beta") {
    xMin = 0.001;
    xMax = 0.999;
  } else if (dist === "uniform") {
    xMin = params.low ?? 0;
    xMax = params.high ?? 1;
  }

  const step = (xMax - xMin) / nPoints;
  const points: Array<{ x: number; y: number }> = [];

  for (let i = 0; i <= nPoints; i++) {
    const x = xMin + i * step;
    let y = 0;

    if (dist === "normal" || dist === "gaussian") {
      y = jStat.normal.pdf(x, params.mu ?? params.loc ?? 0, params.sigma ?? params.scale ?? 1) as number;
    } else if (dist === "halfnormal" || dist === "half_normal") {
      y = halfNormalPdf(x, params.sigma ?? params.scale ?? 1);
    } else if (dist === "gamma") {
      const alpha = params.alpha ?? params.concentration ?? 2;
      const rate = params.beta ?? params.rate ?? 1;
      y = jStat.gamma.pdf(x, alpha, 1 / rate) as number;
    } else if (dist === "beta") {
      y = jStat.beta.pdf(x, params.alpha ?? params.a ?? 2, params.beta ?? params.b ?? 2) as number;
    } else if (dist === "uniform") {
      y =
        x >= (params.low ?? 0) && x <= (params.high ?? 1)
          ? 1 / ((params.high ?? 1) - (params.low ?? 0))
          : 0;
    }

    points.push({ x: Math.round(x * 1000) / 1000, y });
  }

  return points;
}
