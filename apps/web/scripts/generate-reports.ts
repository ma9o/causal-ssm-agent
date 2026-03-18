/**
 * Generate markdown reports for all datasets in data/.
 *
 * Usage:
 *   bun apps/web/scripts/generate-reports.ts            # all datasets
 *   bun apps/web/scripts/generate-reports.ts GOLDEN      # single dataset
 */

import { readdirSync, existsSync, readFileSync, writeFileSync } from "fs";
import { join, resolve } from "path";
import { type AllStageData, generateMarkdown } from "@/lib/utils/generate-markdown";
import type { StageId } from "@causal-ssm/api-types";
import { STAGE_IDS } from "@causal-ssm/api-types";

const DATA_DIR = resolve(import.meta.dirname, "../../../data");

/** Parse JSON that may contain non-standard tokens like Infinity/NaN. */
function parseJson(text: string): unknown {
  // Replace bare Infinity/NaN with null so JSON.parse succeeds.
  // Handles both top-level and escaped (inside JSON strings) occurrences.
  const sanitized = text
    .replace(/(?<=[:,\[]\s*)-?Infinity\b/g, "null")
    .replace(/(?<=[:,\[]\s*)NaN\b/g, "null");
  return JSON.parse(sanitized);
}

/** Unwrap Prefect result envelope if present, otherwise return as-is. */
function unwrapPrefect(raw: unknown): unknown {
  if (
    raw &&
    typeof raw === "object" &&
    "result" in raw &&
    "metadata" in raw
  ) {
    const result = (raw as { result: unknown }).result;
    return typeof result === "string" ? parseJson(result) : result;
  }
  return raw;
}

function loadStageData(runDir: string): AllStageData {
  const data: AllStageData = {};
  for (const stageId of STAGE_IDS) {
    const filePath = join(runDir, `${stageId}.json`);
    if (existsSync(filePath)) {
      const raw = parseJson(readFileSync(filePath, "utf-8"));
      (data as Record<StageId, unknown>)[stageId] = unwrapPrefect(raw);
    }
  }
  return data;
}

function main() {
  const filter = process.argv[2];
  const entries = readdirSync(DATA_DIR, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .filter((d) => !filter || d.name === filter)
    .map((d) => d.name);

  if (entries.length === 0) {
    console.error(filter ? `Dataset "${filter}" not found in ${DATA_DIR}` : "No datasets found");
    process.exit(1);
  }

  for (const name of entries) {
    const runDir = join(DATA_DIR, name, "run");
    if (!existsSync(runDir)) {
      console.warn(`Skipping ${name}: no run/ directory`);
      continue;
    }

    const data = loadStageData(runDir);
    const stageCount = Object.keys(data).length;
    if (stageCount === 0) {
      console.warn(`Skipping ${name}: no stage JSON files`);
      continue;
    }

    try {
      const markdown = generateMarkdown(data, name);
      const outPath = join(DATA_DIR, name, "report.md");
      writeFileSync(outPath, markdown);
      console.log(`${name}: ${stageCount} stages → ${outPath}`);
    } catch (err) {
      console.error(`${name}: failed — ${err instanceof Error ? err.message : err}`);
    }
  }
}

main();
