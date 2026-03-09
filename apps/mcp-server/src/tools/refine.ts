import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { defineTool } from "../define-tool";
import { INTERACTIVE_STAGES, STAGE_IDS, type InteractiveStage } from "../generated/stage-config";
import { getDeploymentId, getFlowRun, triggerRun } from "../services/prefect";
import { readStageResult } from "../services/stage-reader";

const STAGE_ORDER = STAGE_IDS;

interface RefineArgs {
  run_id: string;
  stage: InteractiveStage;
  edits: Record<string, unknown>;
}

export function registerRefineTool(server: McpServer) {
  defineTool(
    server,
    "refine",
    "Edit a completed stage's output and re-run the pipeline from that point. Only interactive stages (1a, 1b, 4) support editing.",
    {
      run_id: z.string(),
      stage: z.enum(INTERACTIVE_STAGES),
      edits: z.record(z.unknown()),
    },
    async (raw) => {
      const { run_id, stage, edits } = raw as unknown as RefineArgs;

      // Read existing stage data
      const existing = await readStageResult(run_id, stage, true);
      if (!existing) {
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({
                error: `Stage ${stage} has no results to refine. Wait for it to complete.`,
              }),
            },
          ],
          isError: true,
        };
      }

      // Merge edits onto existing data, stripping internal-only fields
      const merged = { ...existing, ...edits };
      delete merged.llm_trace;
      delete merged.outcome;

      // Fetch original flow run parameters
      const flowRun = await getFlowRun(run_id);
      const originalParams = flowRun.parameters ?? {};
      const existingOverrides =
        (originalParams.stage_overrides as Record<string, unknown>) ?? {};

      const newParams = {
        ...originalParams,
        stage_overrides: {
          ...existingOverrides,
          [stage]: merged,
        },
      };

      // Trigger new pipeline run
      const deploymentId = await getDeploymentId();
      const newRunId = await triggerRun(deploymentId, newParams);

      const stageIdx = STAGE_ORDER.indexOf(stage);
      const resumesFrom =
        stageIdx + 1 < STAGE_ORDER.length ? STAGE_ORDER[stageIdx + 1] : null;

      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              new_run_id: newRunId,
              edited_stage: stage,
              resumes_from: resumesFrom,
              message: `Stage ${stage} edited. New pipeline run started${resumesFrom ? ` from ${resumesFrom}` : ""}.`,
            }),
          },
        ],
      };
    },
  );
}
