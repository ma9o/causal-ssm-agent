export type TrialCreditStatus = "available" | "unknown" | "exhausted";

type AccessScope = string;

export type AccessStatus =
  | { authScope: AccessScope; mode: "local"; canRun: true }
  | { authScope: AccessScope; mode: "user"; canRun: true }
  | { authScope: AccessScope; mode: "anonymous"; canRun: true; creditStatus: TrialCreditStatus }
  | { authScope: AccessScope; mode: "none"; canRun: false; reason: "anonymous_exhausted" | "local_missing_key" | "misconfigured" };
