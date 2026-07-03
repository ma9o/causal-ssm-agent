type EnvMap = Partial<Record<string, string | undefined>>;

// Runtime URL defaults are intentionally centralized for deployment builds.
function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export const DEFAULT_TOOL_SERVER_URL = "http://localhost:8100";

export function getToolServerUrl(env: EnvMap = process.env): string {
  return trimTrailingSlash(env.TOOL_SERVER_URL ?? DEFAULT_TOOL_SERVER_URL);
}
