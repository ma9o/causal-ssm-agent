import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { defineTool } from "../define-tool";
import { STAGE_IDS, type StageId } from "../generated/stage-config";
import { getFlowRun, getTaskRuns } from "../services/prefect";
import { readStageResult } from "../services/stage-reader";

/** Map Prefect task names to stage IDs. */
const TASK_TO_STAGE: Record<string, StageId> = {
  preprocess_raw_input: "stage-0",
  propose_latent_model: "stage-1a",
  propose_measurement_with_identifiability_fix: "stage-1b",
  populate_indicators: "stage-2",
  validate_extraction: "stage-3",
  stage4_orchestrated_flow: "stage-4",
  stage4b_parametric_id_flow: "stage-4b",
  fit_model: "stage-5",
  run_interventions: "stage-6",
};

type FlowStatus = "pending" | "running" | "completed" | "failed" | "cancelled";

function mapPrefectState(stateType: string): FlowStatus {
  switch (stateType.toUpperCase()) {
    case "COMPLETED":
      return "completed";
    case "FAILED":
    case "CRASHED":
      return "failed";
    case "CANCELLED":
    case "CANCELLING":
      return "cancelled";
    case "RUNNING":
      return "running";
    default:
      return "pending";
  }
}

interface ResultsArgs {
  run_id: string;
  stage?: StageId;
  include_large_arrays?: boolean;
}

export function registerResultsTool(server: McpServer) {
  defineTool(
    server,
    "results",
    "Check pipeline status and retrieve stage outputs. Omit stage for overall status; provide stage for full output.",
    {
      run_id: z.string(),
      stage: z.enum(STAGE_IDS).optional(),
      include_large_arrays: z.boolean().optional(),
    },
    async (raw) => {
      const { run_id, stage, include_large_arrays } = raw as unknown as ResultsArgs;

      // Stage-specific: read and return the JSON
      if (stage) {
        const data = await readStageResult(run_id, stage, include_large_arrays ?? false);
        if (!data) {
          return {
            content: [
              {
                type: "text" as const,
                text: JSON.stringify({
                  status: "not_available",
                  message: `No data for ${stage} yet. The stage may still be running.`,
                }),
              },
            ],
          };
        }
        return {
          content: [{ type: "text" as const, text: JSON.stringify(data) }],
        };
      }

      // Overall status: query Prefect for flow + task states
      const flowRun = await getFlowRun(run_id);
      const status = mapPrefectState(flowRun.state.type);
      const taskRuns = await getTaskRuns(run_id);

      const completed: string[] = [];
      const running: string[] = [];
      const failed: string[] = [];

      for (const task of taskRuns) {
        const stageId = TASK_TO_STAGE[task.name];
        if (!stageId) continue;

        const taskStatus = mapPrefectState(task.state.type);
        if (taskStatus === "completed") completed.push(stageId);
        else if (taskStatus === "running") running.push(stageId);
        else if (taskStatus === "failed") failed.push(stageId);
      }

      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              status,
              completed_stages: completed,
              running_stages: running,
              failed_stages: failed,
            }),
          },
        ],
      };
    },
  );
}
