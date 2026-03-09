// PKCE utilities + localStorage helpers for OpenRouter OAuth

const OPENROUTER_KEY = "openrouter_user_key";
const CODE_VERIFIER_KEY = "openrouter_code_verifier";

// ---------------------------------------------------------------------------
// PKCE
// ---------------------------------------------------------------------------

function base64UrlEncode(buffer: Uint8Array): string {
  const base64 = btoa(String.fromCharCode(...buffer));
  return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/g, "");
}

function generateCodeVerifier(): string {
  const array = new Uint8Array(32);
  crypto.getRandomValues(array);
  return base64UrlEncode(array);
}

async function generateCodeChallenge(verifier: string): Promise<string> {
  const data = new TextEncoder().encode(verifier);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return base64UrlEncode(new Uint8Array(digest));
}

// ---------------------------------------------------------------------------
// localStorage helpers
// ---------------------------------------------------------------------------

export function getUserApiKey(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(OPENROUTER_KEY);
}

export function setUserApiKey(key: string): void {
  localStorage.setItem(OPENROUTER_KEY, key);
}

export function clearUserApiKey(): void {
  localStorage.removeItem(OPENROUTER_KEY);
}

export function getCodeVerifier(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem(CODE_VERIFIER_KEY);
}

export function clearCodeVerifier(): void {
  localStorage.removeItem(CODE_VERIFIER_KEY);
}

// ---------------------------------------------------------------------------
// OAuth flow
// ---------------------------------------------------------------------------

export async function initiateOpenRouterAuth(callbackUrl: string): Promise<void> {
  const verifier = generateCodeVerifier();
  localStorage.setItem(CODE_VERIFIER_KEY, verifier);
  const challenge = await generateCodeChallenge(verifier);

  const url = new URL("https://openrouter.ai/auth");
  url.searchParams.set("callback_url", callbackUrl);
  url.searchParams.set("code_challenge", challenge);
  url.searchParams.set("code_challenge_method", "S256");

  window.location.href = url.toString();
}
