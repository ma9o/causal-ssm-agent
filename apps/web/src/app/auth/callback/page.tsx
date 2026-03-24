"use client";

import { clearCodeVerifier, getCodeVerifier } from "@/lib/auth";
import { getIdentity, setIdentity } from "@/lib/identity";
import { generateAnonymousWorkspaceId } from "@/lib/workspace-id";
import { generateWorkspaceAccessCode } from "@/lib/resume-key";
import { Loader2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { use, useEffect, useRef, useState } from "react";

export default function AuthCallbackPage({
  searchParams,
}: {
  searchParams: Promise<{ code?: string; flow_id?: string }>;
}) {
  const { code, flow_id: flowId } = use(searchParams);
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const startedRef = useRef(false);

  useEffect(() => {
    if (!code || startedRef.current) return;
    startedRef.current = true;

    if (!flowId) {
      setError("Authentication session is missing a flow id. Please try again.");
      return;
    }

    const codeVerifier = getCodeVerifier(flowId);
    if (!codeVerifier) {
      setError("Authentication session expired. Please start the sign-in flow again.");
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
        const existingIdentity = getIdentity();
        setIdentity({
          workspaceId: existingIdentity?.workspaceId ?? generateAnonymousWorkspaceId(),
          accessCode: existingIdentity?.accessCode ?? generateWorkspaceAccessCode(),
          kind: "openrouter",
        });
        router.push("/");
      })
      .catch(() => {
        clearCodeVerifier(flowId);
        setError("Failed to complete authentication. Please start the sign-in flow again.");
      });
  }, [code, flowId, router]);

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
      {error ? (
        <div className="text-center space-y-4">
          <p className="text-sm text-destructive">{error}</p>
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
