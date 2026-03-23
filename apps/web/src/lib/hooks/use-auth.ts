"use client";

import { useCallback, useEffect, useState } from "react";
import { clearUserApiKey, getUserApiKey } from "@/lib/auth";
import { getIdentity, setIdentity, type WorkspaceIdentity } from "@/lib/identity";
import { generateAnonymousWorkspaceId } from "@/lib/workspace-id";
import { generateWorkspaceAccessCode } from "@/lib/resume-key";

export type AuthState = {
  /** User's OpenRouter API key (null if anonymous / signed out) */
  userKey: string | null;
  /** Persistent identity (null until first submit or OAuth) */
  identity: WorkspaceIdentity | null;
  /** Server trial credits available */
  hasCredits: boolean | null;
  /** True when user has no access at all (no key + no trial credits) */
  noAccess: boolean;
  /** Sign out — clears API key but preserves identity */
  signOut: () => void;
  /** Ensure identity exists; creates anonymous one if needed. Returns workspace identity. */
  ensureIdentity: () => WorkspaceIdentity;
};

export function useAuth(): AuthState {
  const [userKey, setUserKey] = useState<string | null>(() =>
    typeof window !== "undefined" ? getUserApiKey() : null
  );
  const [identity, setIdentityState] = useState<WorkspaceIdentity | null>(() =>
    typeof window !== "undefined" ? getIdentity() : null
  );
  const [hasCredits, setHasCredits] = useState<boolean | null>(null);

  useEffect(() => {
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

  const ensureIdentity = useCallback((): WorkspaceIdentity => {
    const existing = getIdentity();
    if (existing) {
      setIdentityState(existing);
      return existing;
    }
    const id: WorkspaceIdentity = {
      workspaceId: generateAnonymousWorkspaceId(),
      accessCode: generateWorkspaceAccessCode(),
      kind: "anonymous",
    };
    setIdentity(id);
    setIdentityState(id);
    return id;
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
