"use client";

import { useCallback, useEffect, useState } from "react";
import { clearUserApiKey, getUserApiKey } from "@/lib/auth";
import { getIdentity, setIdentity, type UserIdentity } from "@/lib/identity";
import { generateAnonymousUserId } from "@/lib/user-id";

export type AuthState = {
  /** User's OpenRouter API key (null if anonymous / signed out) */
  userKey: string | null;
  /** Persistent identity (null until first submit or OAuth) */
  identity: UserIdentity | null;
  /** Server trial credits available */
  hasCredits: boolean | null;
  /** True when user has no access at all (no key + no trial credits) */
  noAccess: boolean;
  /** Sign out — clears API key but preserves identity */
  signOut: () => void;
  /** Ensure identity exists; creates anonymous one if needed. Returns userId. */
  ensureIdentity: () => string;
};

export function useAuth(): AuthState {
  const [userKey, setUserKey] = useState<string | null>(null);
  const [identity, setIdentityState] = useState<UserIdentity | null>(null);
  const [hasCredits, setHasCredits] = useState<boolean | null>(null);

  useEffect(() => {
    setUserKey(getUserApiKey());
    setIdentityState(getIdentity());
    fetch("/api/auth/credits")
      .then((r) => r.json())
      .then((d) => setHasCredits(d.hasCredits))
      .catch(() => setHasCredits(false));
  }, []);

  const signOut = useCallback(() => {
    clearUserApiKey();
    setUserKey(null);
    // Identity is NOT cleared — user can sign back in and recover it
  }, []);

  const ensureIdentity = useCallback((): string => {
    // Already have identity — reuse
    const existing = getIdentity();
    if (existing) {
      setIdentityState(existing);
      return existing.userId;
    }
    // Create anonymous identity
    const id: UserIdentity = { userId: generateAnonymousUserId(), kind: "anonymous" };
    setIdentity(id);
    setIdentityState(id);
    return id.userId;
  }, []);

  return {
    userKey,
    identity,
    hasCredits,
    noAccess: !userKey && hasCredits === false,
    signOut,
    ensureIdentity,
  };
}
