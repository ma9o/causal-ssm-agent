// Persistent user identity — survives sign-out (only API key is cleared)

export type UserIdentity = {
  userId: string;
  kind: "anonymous" | "openrouter";
};

const IDENTITY_KEY = "user_identity";

export function getIdentity(): UserIdentity | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(IDENTITY_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as UserIdentity;
  } catch {
    return null;
  }
}

export function setIdentity(identity: UserIdentity): void {
  localStorage.setItem(IDENTITY_KEY, JSON.stringify(identity));
}

export function clearIdentity(): void {
  localStorage.removeItem(IDENTITY_KEY);
}
