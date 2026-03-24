import { cookies } from "next/headers";
import { getIronSession, type SessionOptions } from "iron-session";
import "@/lib/server/root-env";

const OPENROUTER_SESSION_COOKIE = "openrouter_session";
const OPENROUTER_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30;

export type OpenRouterSession = {
  apiKey: string;
};

type OpenRouterSessionStore = Partial<OpenRouterSession>;

function getOpenRouterSessionSecret(): string | undefined {
  const secret = process.env.OPENROUTER_SESSION_SECRET;
  if (!secret || secret.length < 32) {
    return undefined;
  }
  return secret;
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

export function createOpenRouterSession(apiKey: string): OpenRouterSession {
  return { apiKey };
}

export function hasOpenRouterSessionSecret(): boolean {
  return getSessionOptions() !== null;
}

export async function readOpenRouterSession(): Promise<OpenRouterSession | null> {
  const session = await getOpenRouterSessionStore();
  if (!session || typeof session.apiKey !== "string") {
    return null;
  }

  return {
    apiKey: session.apiKey,
  };
}

export async function writeOpenRouterSession(session: OpenRouterSession): Promise<void> {
  const cookieSession = await getOpenRouterSessionStore();
  if (!cookieSession) {
    throw new Error("OPENROUTER_SESSION_SECRET is not configured");
  }

  cookieSession.apiKey = session.apiKey;
  await cookieSession.save();
}

export async function clearOpenRouterSession(): Promise<void> {
  const session = await getOpenRouterSessionStore();
  if (!session) {
    return;
  }

  session.destroy();
}
