import type { AccessStatus, TrialCreditStatus } from "@/lib/auth-status";
import "@/lib/server/root-env";
import { readOpenRouterSession } from "@/lib/server/openrouter-session";

const OPENROUTER_CREDITS_TTL_MS = 60_000;

type CreditsCacheEntry = {
  creditStatus: TrialCreditStatus;
  ts: number;
  cacheKey: string;
};

type UserAccess = {
  apiKey: string;
  mode: "user";
};

type TrialAccess = {
  apiKey: string;
  creditStatus: TrialCreditStatus;
  mode: "trial";
};

type NoAccess = {
  mode: "none";
  reason: Extract<AccessStatus, { mode: "none" }>["reason"];
};

export type ResolvedOpenRouterAccess = UserAccess | TrialAccess | NoAccess;
export type RunnableOpenRouterAccess = Exclude<ResolvedOpenRouterAccess, NoAccess>;
export type RunnableOpenRouterAccessMode = RunnableOpenRouterAccess["mode"];

let creditsCache: CreditsCacheEntry | null = null;

function getOpenRouterTrialApiKey(): string | undefined {
  return process.env.OPENROUTER_TRIAL_API_KEY;
}

function getOpenRouterCreditsApiKey(): string | undefined {
  return process.env.OPENROUTER_CREDITS_API_KEY;
}

function buildCreditsCacheKey(trialApiKey: string, creditsApiKey: string): string {
  return `${trialApiKey}:${creditsApiKey}`;
}

async function getTrialCreditStatus(trialApiKey: string): Promise<TrialCreditStatus> {
  const creditsApiKey = getOpenRouterCreditsApiKey();
  if (!creditsApiKey) {
    return "unknown";
  }

  const cacheKey = buildCreditsCacheKey(trialApiKey, creditsApiKey);
  if (
    creditsCache &&
    creditsCache.cacheKey === cacheKey &&
    Date.now() - creditsCache.ts < OPENROUTER_CREDITS_TTL_MS
  ) {
    return creditsCache.creditStatus;
  }

  try {
    const response = await fetch("https://openrouter.ai/api/v1/credits", {
      headers: { Authorization: `Bearer ${creditsApiKey}` },
      cache: "no-store",
    });
    if (!response.ok) {
      return "unknown";
    }

    const payload = (await response.json()) as {
      data?: { total_credits?: number; total_usage?: number };
    };
    const totalCredits = payload.data?.total_credits;
    const totalUsage = payload.data?.total_usage;
    if (typeof totalCredits !== "number" || typeof totalUsage !== "number") {
      return "unknown";
    }

    const creditStatus: TrialCreditStatus = totalCredits > totalUsage ? "available" : "exhausted";
    creditsCache = { creditStatus, ts: Date.now(), cacheKey };
    return creditStatus;
  } catch {
    return "unknown";
  }
}

export async function resolveOpenRouterAccess(): Promise<ResolvedOpenRouterAccess> {
  const session = await readOpenRouterSession();
  if (session) {
    return {
      mode: "user",
      apiKey: session.apiKey,
    };
  }

  const trialApiKey = getOpenRouterTrialApiKey();
  if (!trialApiKey) {
    return {
      mode: "none",
      reason: "misconfigured",
    };
  }

  const creditStatus = await getTrialCreditStatus(trialApiKey);
  if (creditStatus === "exhausted") {
    return {
      mode: "none",
      reason: "trial_exhausted",
    };
  }

  return {
    mode: "trial",
    apiKey: trialApiKey,
    creditStatus,
  };
}

export function toAccessStatus(access: ResolvedOpenRouterAccess): AccessStatus {
  switch (access.mode) {
    case "user":
      return { mode: "user", canRun: true };
    case "trial":
      return { mode: "trial", canRun: true, creditStatus: access.creditStatus };
    case "none":
      return { mode: "none", canRun: false, reason: access.reason };
  }
}

export async function getOpenRouterStatus(): Promise<AccessStatus> {
  return toAccessStatus(await resolveOpenRouterAccess());
}
