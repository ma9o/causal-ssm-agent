// PKCE utilities + sessionStorage helpers for OpenRouter OAuth

const CODE_VERIFIER_KEY_PREFIX = "openrouter_code_verifier:";

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

function getCodeVerifierStorageKey(flowId: string): string {
  return `${CODE_VERIFIER_KEY_PREFIX}${flowId}`;
}

function isCodeVerifierStorageKey(key: string): boolean {
  return key.startsWith(CODE_VERIFIER_KEY_PREFIX);
}

function pruneCodeVerifiers(exceptFlowId?: string): void {
  if (typeof window === "undefined") return;

  const keepKey = exceptFlowId ? getCodeVerifierStorageKey(exceptFlowId) : null;
  const keysToDelete: string[] = [];

  for (let index = 0; index < sessionStorage.length; index += 1) {
    const key = sessionStorage.key(index);
    if (!key || !isCodeVerifierStorageKey(key) || key === keepKey) {
      continue;
    }
    keysToDelete.push(key);
  }

  for (const key of keysToDelete) {
    sessionStorage.removeItem(key);
  }
}

export function getCodeVerifier(flowId: string): string | null {
  if (typeof window === "undefined") return null;
  return sessionStorage.getItem(getCodeVerifierStorageKey(flowId));
}

export function clearCodeVerifier(flowId: string): void {
  sessionStorage.removeItem(getCodeVerifierStorageKey(flowId));
}

function generateFlowId(): string {
  return crypto.randomUUID();
}

// ---------------------------------------------------------------------------
// OAuth flow
// ---------------------------------------------------------------------------

export async function initiateOpenRouterAuth(callbackUrl: string): Promise<void> {
  const flowId = generateFlowId();
  pruneCodeVerifiers(flowId);
  const verifier = generateCodeVerifier();
  sessionStorage.setItem(getCodeVerifierStorageKey(flowId), verifier);
  const challenge = await generateCodeChallenge(verifier);
  const callback = new URL(callbackUrl);
  callback.searchParams.set("flow_id", flowId);

  const url = new URL("https://openrouter.ai/auth");
  url.searchParams.set("callback_url", callback.toString());
  url.searchParams.set("code_challenge", challenge);
  url.searchParams.set("code_challenge_method", "S256");

  window.location.href = url.toString();
}
