import { loadEnvConfig } from "@next/env";
import { join } from "node:path";

let rootEnvLoaded = false;

function ensureRootEnvLoaded() {
  if (rootEnvLoaded) {
    return;
  }

  const projectDir = process.cwd();
  loadEnvConfig(projectDir);
  loadEnvConfig(join(projectDir, "..", ".."));
  rootEnvLoaded = true;
}

ensureRootEnvLoaded();

export {};
