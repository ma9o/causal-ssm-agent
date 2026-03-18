type EnvMap = Partial<Record<string, string | undefined>>;

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export const DEFAULT_PREFECT_API_URL = "http://localhost:4200/api";
export const DEFAULT_TOOL_SERVER_URL = "http://localhost:8100";
export const DEFAULT_PREFECT_EVENTS_URL = "ws://localhost:4200/api/events/out";

export function getPrefectApiUrl(env: EnvMap = process.env): string {
  return trimTrailingSlash(env.PREFECT_API_URL ?? DEFAULT_PREFECT_API_URL);
}

export function getToolServerUrl(env: EnvMap = process.env): string {
  return trimTrailingSlash(env.TOOL_SERVER_URL ?? DEFAULT_TOOL_SERVER_URL);
}

export function getPrefectEventsUrl(
  origin: string,
  env: EnvMap = process.env,
): string {
  const configured = env.NEXT_PUBLIC_PREFECT_EVENTS_URL;
  if (configured) {
    return trimTrailingSlash(configured);
  }

  if (env.NODE_ENV === "development") {
    return DEFAULT_PREFECT_EVENTS_URL;
  }

  return `${origin.replace(/^http/, "ws")}/prefect/events/out`;
}
