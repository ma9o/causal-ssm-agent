import { NextResponse } from "next/server";
import {
  createOpenRouterSession,
  hasOpenRouterSessionSecret,
  writeOpenRouterSession,
} from "@/lib/server/openrouter-session";
import { authorizeWorkspacesForOpenRouterUser } from "@/lib/server/workspace-ownership";
import {
  clearAuthorizedWorkspaceIds,
  readAuthorizedWorkspaceIds,
} from "@/lib/server/workspace-session";

export async function POST(request: Request) {
  if (process.env.DEPLOYMENT_ENV !== "production") {
    return NextResponse.json(
      { error: "OpenRouter sign-in is only available in production." },
      { status: 403 },
    );
  }

  const { code, code_verifier } = await request.json();

  if (!code) {
    return NextResponse.json({ error: "Missing authorization code" }, { status: 400 });
  }
  if (!code_verifier) {
    return NextResponse.json({ error: "Missing PKCE code verifier" }, { status: 400 });
  }
  if (!hasOpenRouterSessionSecret()) {
    return NextResponse.json(
      { error: "APP_SECRET is not configured" },
      { status: 500 },
    );
  }

  try {
    const res = await fetch("https://openrouter.ai/api/v1/auth/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code,
        code_verifier,
        code_challenge_method: "S256",
      }),
    });

    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      return NextResponse.json(
        { error: body.error?.message || "Failed to exchange code" },
        { status: res.status },
      );
    }

    const { key, user_id } = await res.json();
    if (typeof key !== "string" || typeof user_id !== "string") {
      return NextResponse.json(
        { error: "OpenRouter did not return a usable account session." },
        { status: 502 },
      );
    }

    const sessionWorkspaceIds = await readAuthorizedWorkspaceIds();
    await authorizeWorkspacesForOpenRouterUser(user_id, sessionWorkspaceIds);
    await writeOpenRouterSession(createOpenRouterSession(key, user_id));
    await clearAuthorizedWorkspaceIds();
    return NextResponse.json({ ok: true });
  } catch {
    return NextResponse.json({ error: "Failed to exchange code" }, { status: 500 });
  }
}
