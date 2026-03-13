import { NextResponse } from "next/server";
import { getDefaultApiKey } from "@/lib/api/resolve-api-key";

// Simple in-memory cache to avoid hammering OpenRouter on every page load
let cache: { hasCredits: boolean; ts: number } | null = null;
const TTL = 60_000; // 60 seconds

export async function GET() {
  const apiKey = getDefaultApiKey();
  if (!apiKey) {
    return NextResponse.json({ hasCredits: false });
  }

  if (cache && Date.now() - cache.ts < TTL) {
    return NextResponse.json({ hasCredits: cache.hasCredits });
  }

  try {
    const res = await fetch("https://openrouter.ai/api/v1/credits", {
      headers: { Authorization: `Bearer ${apiKey}` },
    });

    if (!res.ok) {
      return NextResponse.json({ hasCredits: false });
    }

    const { data } = await res.json();
    const hasCredits = data.total_credits > data.total_usage;
    cache = { hasCredits, ts: Date.now() };
    return NextResponse.json({ hasCredits });
  } catch {
    return NextResponse.json({ hasCredits: false });
  }
}
