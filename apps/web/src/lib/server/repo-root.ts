import { existsSync } from "node:fs";
import { join } from "node:path";

export function getRepoRoot(): string {
  const cwd = process.cwd();
  const candidates = [cwd, join(cwd, ".."), join(cwd, "..", "..")];

  for (const candidate of candidates) {
    if (existsSync(join(candidate, "apps")) && existsSync(join(candidate, "packages"))) {
      return candidate;
    }
  }

  return join(cwd, "..", "..");
}
