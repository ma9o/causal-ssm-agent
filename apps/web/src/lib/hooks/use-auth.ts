"use client";

import { useCallback, useEffect, useState } from "react";
import type { AccessStatus } from "@/lib/auth-status";

export type AuthState = {
  /** Server-derived OpenRouter access status */
  access: AccessStatus | null;
  /** True when user has no access at all (no key + no trial credits) */
  noAccess: boolean;
  /** Sign out — clears the server session */
  signOut: () => Promise<void>;
};

export function useAuth(): AuthState {
  const [access, setAccess] = useState<AccessStatus | null>(null);

  const refresh = useCallback(async () => {
    try {
      const response = await fetch("/api/auth/status", { cache: "no-store" });
      if (!response.ok) {
        throw new Error("Failed to load auth status");
      }
      setAccess((await response.json()) as AccessStatus);
    } catch {
      setAccess(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const signOut = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
    } finally {
      await refresh();
    }
  }, [refresh]);

  return {
    access,
    noAccess: access?.canRun === false,
    signOut,
  };
}
