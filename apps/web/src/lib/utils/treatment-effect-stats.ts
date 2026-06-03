/**
 * Pure numeric reductions over a {@link TreatmentEffect}'s posterior draws and
 * temporal decomposition. These are the single source for the summary statistics
 * surfaced by both the treatment-ranking table and the Stage 6 scenario rail —
 * no statistical logic is duplicated across the frontend.
 */

import type { TreatmentEffect } from "@nof1-causal-lab/api-types";
import { CI_LOWER, CI_UPPER } from "@/lib/constants/diagnostics";
import { quantile } from "@/lib/utils/histogram";

/** Mean of posterior draws, or null if unavailable. */
export function meanDraws(draws: number[] | null | undefined): number | null {
  if (!draws || draws.length === 0) return null;
  return draws.reduce((a, b) => a + b, 0) / draws.length;
}

/** Posterior probability that the effect is positive, or null if unavailable. */
export function probPositive(draws: number[] | null | undefined): number | null {
  if (!draws || draws.length === 0) return null;
  return draws.filter((draw) => draw > 0).length / draws.length;
}

/** 95% credible interval (2.5th–97.5th percentile) of posterior draws. */
export function drawsCI(
  draws: number[] | null | undefined,
): { lower: number; upper: number } | null {
  if (!draws || draws.length === 0) return null;
  const sorted = [...draws].sort((a, b) => a - b);
  return { lower: quantile(sorted, CI_LOWER), upper: quantile(sorted, CI_UPPER) };
}

/** Maximum absolute forward-simulated effect, or null if no temporal decomposition. */
export function peakEffect(effect: TreatmentEffect): number | null {
  return effect.temporal?.peak_effect ?? null;
}

/** Days from intervention onset to the peak effect, or null. */
export function timeToPeak(effect: TreatmentEffect): number | null {
  return effect.temporal?.time_to_peak_days ?? null;
}
