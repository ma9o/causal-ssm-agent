/**
 * Generate MCP server constants from pipeline metadata.
 *
 * Reads mcp-meta.json (produced by Python export_schemas.py) and generates
 * typed constants for tool schemas and stage configuration.
 *
 * Usage:
 *   cd apps/mcp-server
 *   bun run scripts/generate.ts
 */

import { readFileSync, writeFileSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";

const ROOT = dirname(dirname(resolve(import.meta.filename)));
const SCHEMAS_DIR = resolve(ROOT, "..", "..", "packages", "api-types", "schemas");
const META_PATH = resolve(SCHEMAS_DIR, "mcp-meta.json");
const OUTPUT_DIR = resolve(ROOT, "src", "generated");

interface McpMeta {
  stages: Array<{ id: string; interactive: boolean }>;
  interactive_stages: string[];
  large_array_fields: Record<string, string[]>;
  large_nested_fields: Record<string, Record<string, string[]>>;
}

function main() {
  const meta: McpMeta = JSON.parse(readFileSync(META_PATH, "utf-8"));

  const stageIds = meta.stages.map((s) => s.id);
  const interactiveStages = meta.interactive_stages;

  const code = `/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python pipeline metadata via:
 *   cd apps/data-pipeline && uv run python scripts/export_schemas.py
 *   cd apps/mcp-server && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/causal_ssm_agent/flows/stages/mcp_meta.py
 */

export const STAGE_IDS = ${JSON.stringify(stageIds)} as const;
export type StageId = (typeof STAGE_IDS)[number];

export const INTERACTIVE_STAGES = ${JSON.stringify(interactiveStages)} as const;
export type InteractiveStage = (typeof INTERACTIVE_STAGES)[number];

export const LARGE_ARRAY_FIELDS: Partial<Record<StageId, string[]>> = ${JSON.stringify(meta.large_array_fields, null, 2)};

export const LARGE_NESTED_FIELDS: Partial<Record<StageId, Record<string, string[]>>> = ${JSON.stringify(meta.large_nested_fields, null, 2)};
`;

  mkdirSync(OUTPUT_DIR, { recursive: true });
  const outPath = resolve(OUTPUT_DIR, "stage-config.ts");
  writeFileSync(outPath, code);
  console.log(`Generated stage config → ${outPath}`);
}

main();
