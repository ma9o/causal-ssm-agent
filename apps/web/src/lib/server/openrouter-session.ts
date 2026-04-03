import { cookies } from "next/headers";
import { getIronSession, type SessionOptions } from "iron-session";
import "@/lib/server/root-env";
import { deriveAppSecret } from "@/lib/server/app-secret";

const OPENROUTER_SESSION_COOKIE = "openrouter_session";
const OPENROUTER_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30;

export type OpenRouterSession = {
  apiKey: string;
  userId: string;
};

type OpenRouterSessionStore = Partial<OpenRouterSession>;

function getOpenRouterSessionSecret(): string | undefined {
  try {
    return deriveAppSecret("openrouter-session");
  } catch {
    return undefined;
  }
}

function getSessionOptions(): SessionOptions | null {
  const password = getOpenRouterSessionSecret();
  if (!password) {
    return null;
  }

  return {
    password,
    cookieName: OPENROUTER_SESSION_COOKIE,
    ttl: OPENROUTER_SESSION_MAX_AGE_SECONDS,
    cookieOptions: {
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      path: "/",
    },
  };
}

async function getOpenRouterSessionStore() {
  const options = getSessionOptions();
  if (!options) {
    return null;
  }

  return getIronSession<OpenRouterSessionStore>(await cookies(), options);
}

export function createOpenRouterSession(apiKey: string, userId: string): OpenRouterSession {
  return { apiKey, userId };
}

export function hasOpenRouterSessionSecret(): boolean {
  return getSessionOptions() !== null;
}

export async function readOpenRouterSession(): Promise<OpenRouterSession | null> {
  const session = await getOpenRouterSessionStore();
  if (!session || typeof session.apiKey !== "string" || typeof session.userId !== "string") {
    return null;
  }

  return {
    apiKey: session.apiKey,
    userId: session.userId,
  };
}

export async function writeOpenRouterSession(session: OpenRouterSession): Promise<void> {
  const cookieSession = await getOpenRouterSessionStore();
  if (!cookieSession) {
    throw new Error("APP_SECRET is not configured");
  }

  cookieSession.apiKey = session.apiKey;
  cookieSession.userId = session.userId;
  await cookieSession.save();
}

export async function clearOpenRouterSession(): Promise<void> {
  const session = await getOpenRouterSessionStore();
  if (!session) {
    return;
  }

  session.destroy();
}
