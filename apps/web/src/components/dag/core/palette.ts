/**
 * Color language for the bespoke DAG renderer, matching the analysis "Living DAG"
 * design playground exactly. The one axis nodes & edges read on is pos/neg:
 * teal = above set-point / positive push, red = below / negative, calm neutral
 * near zero. Hex values are the playground's, kept literal for pixel fidelity.
 */
export const DAG_COLORS = {
  /** Positive / above set-point (teal). */
  positive: "#0f9b8e",
  /** Negative / below set-point (red). */
  negative: "#e0607e",
  /** Near-zero sign / neutral edge. */
  neutral: "#cfd4db",
  /** Reference series, muted text, realized dots. */
  muted: "#9aa0a8",
  /** A latent's single baseline line when no intervention has moved it. */
  slate: "#5b6470",
  /** do() intervention accent (blue). */
  intervention: "#2f6bf0",
  /** Contemporaneous structural edge (static DAG). */
  contemporaneous: "#5b6470",
  /** Lagged (t−1) structural edge (static DAG). */
  lagged: "#9aa0a8",
  /** Counterfactual 95% CrI fill. */
  tealSoft: "rgba(15,155,142,.16)",
  /** Baseline 95% CrI fill. */
  mutedSoft: "rgba(154,160,168,.28)",
  /** do() vertical-marker soft fill. */
  blueSoft: "rgba(47,107,240,.12)",
  /** Primary ink. */
  ink: "#16191d",
  /** Hairlines / borders. */
  line: "#e4e7ec",
  line2: "#cfd4db",
  /** Alternate causal-layer column band. */
  col: "#f6f8fa",
  /** Pruned (severed) edge. */
  pruned: "#c5ccd6",
  /** Severed-edge scissors glyph. */
  scissors: "#9aa7b4",
  /** Realized factual dots. */
  realized: "#7b818b",
} as const;

/**
 * Map a signed value to the pos/neg color axis. `eps` keeps near-zero values
 * calm (neutral) instead of flickering between teal and red.
 */
export function signColor(value: number, eps = 0.012): string {
  if (value > eps) return DAG_COLORS.positive;
  if (value < -eps) return DAG_COLORS.negative;
  return DAG_COLORS.neutral;
}
