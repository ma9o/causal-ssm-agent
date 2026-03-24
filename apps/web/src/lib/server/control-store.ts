import { createClient, type Client } from "@libsql/client";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, isAbsolute, join } from "node:path";
import "@/lib/server/root-env";

const REPO_ROOT = resolveRepoRoot();
const DEFAULT_CONTROL_STORE_PATH = join(REPO_ROOT, ".local", "byok-secret-store.db");
const DEFAULT_CONTROL_STORE_URL = `file:${DEFAULT_CONTROL_STORE_PATH}`;

type ControlStoreConfig = {
  authToken?: string;
  filePath?: string;
  url: string;
};

function resolveRepoRoot(): string {
  const cwd = process.cwd();
  const candidates = [cwd, join(cwd, ".."), join(cwd, "..", "..")];

  for (const candidate of candidates) {
    if (existsSync(join(candidate, "apps")) && existsSync(join(candidate, "packages"))) {
      return candidate;
    }
  }

  return join(cwd, "..", "..");
}

function getControlStoreAuthToken(): string | undefined {
  const configured = process.env.BYOK_SECRET_STORE_AUTH_TOKEN?.trim();
  return configured || undefined;
}

function getControlStoreUrl(): string {
  const configured = process.env.BYOK_SECRET_STORE_URL?.trim();
  return configured || DEFAULT_CONTROL_STORE_URL;
}

function getControlStoreConfig(): ControlStoreConfig {
  const rawUrl = getControlStoreUrl();
  const authToken = getControlStoreAuthToken();

  if (rawUrl.startsWith("file:")) {
    const rawPath = rawUrl.slice("file:".length);
    const filePath = isAbsolute(rawPath) ? rawPath : join(REPO_ROOT, rawPath);
    return {
      filePath,
      url: `file:${filePath}`,
    };
  }

  return {
    url: rawUrl,
    ...(authToken ? { authToken } : {}),
  };
}

export function createControlStoreClient(): Client {
  const config = getControlStoreConfig();
  if (config.filePath) {
    mkdirSync(dirname(config.filePath), { recursive: true, mode: 0o700 });
  }

  return createClient({
    url: config.url,
    ...(config.authToken ? { authToken: config.authToken } : {}),
  });
}
