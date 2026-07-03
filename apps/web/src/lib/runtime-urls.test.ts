import { describe, expect, it } from "vitest";
import { DEFAULT_TOOL_SERVER_URL, getToolServerUrl } from "./runtime-urls";

describe("runtime-urls", () => {
  it("uses the local default when env vars are absent", () => {
    expect(getToolServerUrl({})).toBe(DEFAULT_TOOL_SERVER_URL);
  });

  it("uses the configured URL and trims trailing slashes", () => {
    expect(getToolServerUrl({ TOOL_SERVER_URL: "https://tools.example/" })).toBe(
      "https://tools.example",
    );
  });
});
