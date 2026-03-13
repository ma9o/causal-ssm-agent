"use client";

import { clearCodeVerifier, getCodeVerifier, setUserApiKey } from "@/lib/auth";
import { setIdentity } from "@/lib/identity";
import { generateSessionCode } from "@/lib/session-code";
import { Loader2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { use, useEffect, useState } from "react";

export default function AuthCallbackPage({
  searchParams,
}: {
  searchParams: Promise<{ code?: string }>;
}) {
  const { code } = use(searchParams);
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!code) {
      setError("No authorization code received from OpenRouter.");
      return;
    }

    const codeVerifier = getCodeVerifier();
    clearCodeVerifier();

    fetch("/api/auth/exchange", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ code, code_verifier: codeVerifier }),
    })
      .then((res) => {
        if (!res.ok) throw new Error("Exchange failed");
        return res.json();
      })
      .then(({ key, user_id }) => {
        setUserApiKey(key);
        setIdentity({
          userId: user_id ?? generateSessionCode(),
          kind: "openrouter",
        });
        router.push("/");
      })
      .catch(() => {
        setError("Failed to complete authentication. Please try again.");
      });
  }, [code, router]);

  return (
    <div className="flex min-h-screen items-center justify-center p-4">
      {error ? (
        <div className="text-center space-y-4">
          <p className="text-sm text-destructive">{error}</p>
          <a href="/" className="text-sm text-primary underline underline-offset-2">
            Return home
          </a>
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
