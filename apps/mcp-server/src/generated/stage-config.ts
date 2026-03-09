/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python pipeline metadata via:
 *   cd apps/data-pipeline && uv run python scripts/export_schemas.py
 *   cd apps/mcp-server && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/causal_ssm_agent/flows/stages/mcp_meta.py
 */

export const STAGE_IDS = ["stage-0","stage-1a","stage-1b","stage-2","stage-3","stage-4","stage-4b","stage-5","stage-6"] as const;
export type StageId = (typeof STAGE_IDS)[number];

export const INTERACTIVE_STAGES = ["stage-1a","stage-1b","stage-4"] as const;
export type InteractiveStage = (typeof INTERACTIVE_STAGES)[number];

export const LARGE_ARRAY_FIELDS: Partial<Record<StageId, string[]>> = {
  "stage-4": [
    "prior_predictive_samples"
  ],
  "stage-5": [
    "posterior_marginals",
    "posterior_pairs"
  ]
};

export const LARGE_NESTED_FIELDS: Partial<Record<StageId, Record<string, string[]>>> = {
  "stage-6": {
    "intervention_results": [
      "posterior_draws"
    ]
  }
};
