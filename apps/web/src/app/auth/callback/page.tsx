"use client";

import { clearCodeVerifier, getCodeVerifier } from "@/lib/auth";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useEffect, useMemo, useRef, useState } from "react";

export default function AuthCallbackPage({
  searchParams,
}: {
  searchParams: Promise<{ code?: string; flow_id?: string }>;
}) {
  const { code, flow_id: flowId } = use(searchParams);
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const startedRef = useRef(false);
  const validationError = useMemo(() => {
    if (!code) {
      return null;
    }
    if (!flowId) {
      return "Authentication session is missing a flow id. Please try again.";
    }
    if (!getCodeVerifier(flowId)) {
      return "Authentication session expired. Please start the sign-in flow again.";
    }
    return null;
  }, [code, flowId]);

  useEffect(() => {
    if (!code || !flowId || validationError || startedRef.current) return;
    startedRef.current = true;
    const codeVerifier = getCodeVerifier(flowId);
    if (!codeVerifier) {
      return;
    }

    fetch("/api/auth/exchange", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, code_verifier: codeVerifier }),
    })
      .then((res) => {
        if (!res.ok) throw new Error("Exchange failed");
        clearCodeVerifier(flowId);
        router.push("/");
      })
      .catch((err) => {
        console.error("Auth exchange failed:", err);
        clearCodeVerifier(flowId);
        setError("Failed to complete authentication. Please start the sign-in flow again.");
      });
  }, [code, flowId, router, validationError]);

  if (!code) {
    return (
      <div className="flex min-h-screen items-center justify-center p-4">
        <div className="text-center space-y-4">
          <p className="text-sm text-destructive">No authorization code received from OpenRouter.</p>
          <Link href="/" className="text-sm text-primary underline underline-offset-2">
            Return home
          </Link>
        </div>
      </div>
    );
  }

  return (
      <div className="flex min-h-screen items-center justify-center p-4">
      {error ?? validationError ? (
        <div className="text-center space-y-4">
          <p className="text-sm text-destructive">{error ?? validationError}</p>
          <Link href="/" className="text-sm text-primary underline underline-offset-2">
            Return home
          </Link>
        </div>
      ) : (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          <p className="text-sm">Completing authentication...</p>
        </div>
      )}
    </div>
  );
}
