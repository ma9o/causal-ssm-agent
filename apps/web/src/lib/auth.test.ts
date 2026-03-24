import { afterEach, describe, expect, it, vi } from "vitest";
import { getCodeVerifier, initiateOpenRouterAuth } from "./auth";

class MemoryStorage implements Storage {
  private readonly values = new Map<string, string>();

  get length(): number {
    return this.values.size;
  }

  clear(): void {
    this.values.clear();
  }

  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }

  key(index: number): string | null {
    return [...this.values.keys()][index] ?? null;
  }

  removeItem(key: string): void {
    this.values.delete(key);
  }

  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

const originalWindow = globalThis.window;
const originalSessionStorage = globalThis.sessionStorage;

function restoreGlobal<K extends "window" | "sessionStorage">(
  key: K,
  value: (typeof globalThis)[K],
) {
  if (value === undefined) {
    delete (globalThis as typeof globalThis & Partial<typeof globalThis>)[key];
    return;
  }

  Object.defineProperty(globalThis, key, {
    configurable: true,
    value,
  });
}

describe("auth PKCE storage", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    restoreGlobal("window", originalWindow);
    restoreGlobal("sessionStorage", originalSessionStorage);
  });

  it("prunes stale verifier entries before starting a new auth flow", async () => {
    const storage = new MemoryStorage();
    storage.setItem("openrouter_code_verifier:stale-flow", "stale-verifier");
    storage.setItem("unrelated", "keep-me");

    Object.defineProperty(globalThis, "sessionStorage", {
      configurable: true,
      value: storage,
    });
    Object.defineProperty(globalThis, "window", {
      configurable: true,
      value: {
        location: {
          href: "",
        },
      } as unknown as Window & typeof globalThis,
    });

    vi.spyOn(globalThis.crypto, "randomUUID").mockReturnValue("fresh-flow");

    await initiateOpenRouterAuth("http://localhost/auth/callback");

    expect(storage.getItem("openrouter_code_verifier:stale-flow")).toBeNull();
    expect(storage.getItem("unrelated")).toBe("keep-me");
    expect(getCodeVerifier("fresh-flow")).toBeTypeOf("string");
    const redirectUrl = new URL(
      (globalThis.window as Window & typeof globalThis).location.href,
    );
    const callbackUrl = new URL(
      redirectUrl.searchParams.get("callback_url") ?? "",
    );
    expect(callbackUrl.searchParams.get("flow_id")).toBe("fresh-flow");
  });
});
