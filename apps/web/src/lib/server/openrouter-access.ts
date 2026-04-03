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
  userId: string;
};

type LocalAccess = {
  apiKey: string;
  mode: "local";
};

type AnonymousAccess = {
  apiKey: string;
  creditStatus: TrialCreditStatus;
  mode: "anonymous";
};

type NoAccess = {
  mode: "none";
  reason: Extract<AccessStatus, { mode: "none" }>["reason"];
};

export type ResolvedOpenRouterAccess =
  | UserAccess
  | LocalAccess
  | AnonymousAccess
  | NoAccess;
export type RunnableOpenRouterAccess = Exclude<ResolvedOpenRouterAccess, NoAccess>;
export type RunnableOpenRouterAccessMode = RunnableOpenRouterAccess["mode"];

let creditsCache: CreditsCacheEntry | null = null;

function getOpenRouterApiKey(): string | undefined {
  return process.env.OPENROUTER_API_KEY;
}

function getOpenRouterCreditsApiKey(): string | undefined {
  return process.env.OPENROUTER_CREDITS_API_KEY;
}

function buildCreditsCacheKey(apiKey: string, creditsApiKey: string): string {
  return `${apiKey}:${creditsApiKey}`;
}

async function getTrialCreditStatus(apiKey: string): Promise<TrialCreditStatus> {
  const creditsApiKey = getOpenRouterCreditsApiKey();
  if (!creditsApiKey) {
    return "unknown";
  }

  const cacheKey = buildCreditsCacheKey(apiKey, creditsApiKey);
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
  } catch (e) {
    console.warn("Failed to fetch OpenRouter credit status:", e);
    return "unknown";
  }
}

export async function resolveOpenRouterAccess(): Promise<ResolvedOpenRouterAccess> {
  const apiKey = getOpenRouterApiKey();

  if (process.env.DEPLOYMENT_ENV !== "production") {
    if (!apiKey) {
      return {
        mode: "none",
        reason: "local_missing_key",
      };
    }

    return {
      mode: "local",
      apiKey,
    };
  }

  const session = await readOpenRouterSession();
  if (session) {
    return {
      mode: "user",
      apiKey: session.apiKey,
      userId: session.userId,
    };
  }

  if (!apiKey) {
    return {
      mode: "none",
      reason: "misconfigured",
    };
  }

  const creditStatus = await getTrialCreditStatus(apiKey);
  if (creditStatus === "exhausted") {
    return {
      mode: "none",
      reason: "anonymous_exhausted",
    };
  }

  return {
    mode: "anonymous",
    apiKey,
    creditStatus,
  };
}

export function toAccessStatus(access: ResolvedOpenRouterAccess): AccessStatus {
  switch (access.mode) {
    case "local":
      return { mode: "local", canRun: true };
    case "user":
      return { mode: "user", canRun: true };
    case "anonymous":
      return { mode: "anonymous", canRun: true, creditStatus: access.creditStatus };
    case "none":
      return { mode: "none", canRun: false, reason: access.reason };
  }
}

export function noAccessMessage(reason: NoAccess["reason"]): string {
  switch (reason) {
    case "anonymous_exhausted":
      return "Anonymous credits exhausted. Sign in with OpenRouter to continue.";
    case "local_missing_key":
      return "Local mode requires OPENROUTER_API_KEY to be configured.";
    case "misconfigured":
      return "No OpenRouter access is configured.";
  }
}

export async function getOpenRouterStatus(): Promise<AccessStatus> {
  return toAccessStatus(await resolveOpenRouterAccess());
}
