import { loadEnvConfig } from "@next/env";
import { getRepoRoot } from "@/lib/server/repo-root";

let rootEnvLoaded = false;

function ensureRootEnvLoaded() {
  if (rootEnvLoaded) {
    return;
  }

  loadEnvConfig(getRepoRoot());
  rootEnvLoaded = true;
}

ensureRootEnvLoaded();

export {};
