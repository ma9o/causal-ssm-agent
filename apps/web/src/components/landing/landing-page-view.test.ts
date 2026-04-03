import { describe, expect, it } from "vitest";
import type { AccessStatus } from "@/lib/auth-status";
import { canOfferOpenRouterSignIn } from "./landing-page-view";

describe("canOfferOpenRouterSignIn", () => {
  it("hides OpenRouter sign-in in local mode", () => {
    expect(canOfferOpenRouterSignIn({ mode: "local", canRun: true })).toBe(false);
  });

  it("hides OpenRouter sign-in when local mode is misconfigured", () => {
    const access: AccessStatus = {
      mode: "none",
      canRun: false,
      reason: "local_missing_key",
    };
    expect(canOfferOpenRouterSignIn(access)).toBe(false);
  });

  it("shows OpenRouter sign-in for anonymous mode", () => {
    const access: AccessStatus = {
      mode: "anonymous",
      canRun: true,
      creditStatus: "available",
    };
    expect(canOfferOpenRouterSignIn(access)).toBe(true);
  });
});
