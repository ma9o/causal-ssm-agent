import { describe, expect, it } from "vitest";
import {
  DEFAULT_PREFECT_API_URL,
  DEFAULT_PREFECT_EVENTS_URL,
  DEFAULT_TOOL_SERVER_URL,
  getPrefectApiUrl,
  getPrefectEventsUrl,
  getToolServerUrl,
} from "./runtime-urls";

describe("runtime-urls", () => {
  it("uses local defaults when env vars are absent", () => {
    expect(getPrefectApiUrl({})).toBe(DEFAULT_PREFECT_API_URL);
    expect(getToolServerUrl({})).toBe(DEFAULT_TOOL_SERVER_URL);
    expect(getPrefectEventsUrl("http://example.test", { NODE_ENV: "development" })).toBe(
      DEFAULT_PREFECT_EVENTS_URL,
    );
  });

  it("uses configured server-side URLs and trims trailing slashes", () => {
    expect(getPrefectApiUrl({ PREFECT_API_URL: "https://prefect.example/api/" })).toBe(
      "https://prefect.example/api",
    );
    expect(getToolServerUrl({ TOOL_SERVER_URL: "https://tools.example/" })).toBe(
      "https://tools.example",
    );
  });

  it("prefers configured public websocket URL", () => {
    expect(
      getPrefectEventsUrl("https://app.example", {
        NODE_ENV: "production",
        NEXT_PUBLIC_PREFECT_EVENTS_URL: "wss://prefect.example/events/out/",
      }),
    ).toBe("wss://prefect.example/events/out");
  });

  it("falls back to the app origin for production websocket traffic", () => {
    expect(getPrefectEventsUrl("https://app.example", { NODE_ENV: "production" })).toBe(
      "wss://app.example/prefect/events/out",
    );
  });
});
