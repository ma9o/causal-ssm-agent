export type TrialCreditStatus = "available" | "unknown" | "exhausted";

export type AccessStatus =
  | { mode: "local"; canRun: true }
  | { mode: "user"; canRun: true }
  | { mode: "anonymous"; canRun: true; creditStatus: TrialCreditStatus }
  | { mode: "none"; canRun: false; reason: "anonymous_exhausted" | "local_missing_key" | "misconfigured" };
