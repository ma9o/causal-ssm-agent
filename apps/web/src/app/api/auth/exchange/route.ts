import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const { code, code_verifier } = await request.json();

  if (!code) {
    return NextResponse.json({ error: "Missing authorization code" }, { status: 400 });
  }

  try {
    const res = await fetch("https://openrouter.ai/api/v1/auth/keys", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code,
        ...(code_verifier && {
          code_verifier,
          code_challenge_method: "S256",
        }),
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
    return NextResponse.json({ key, user_id: user_id ?? null });
  } catch {
    return NextResponse.json({ error: "Failed to exchange code" }, { status: 500 });
  }
}
