export type TrialCreditStatus = "available" | "unknown" | "exhausted";

export type AccessStatus =
  | { mode: "user"; canRun: true }
  | { mode: "trial"; canRun: true; creditStatus: TrialCreditStatus }
  | { mode: "none"; canRun: false; reason: "trial_exhausted" | "misconfigured" };
