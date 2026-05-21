import { INTERACTIVE_STAGES, STAGE_TOOLS } from "@nof1-causal-lab/api-types";
import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { createOpenRouter } from "@openrouter/ai-sdk-provider";
import {
  addToolInputExamplesMiddleware,
  convertToModelMessages,
  jsonSchema,
  stepCountIs,
  streamText,
  tool,
  wrapLanguageModel,
} from "ai";
import { NextResponse } from "next/server";

import { getToolServerUrl } from "@/lib/runtime-urls";
import { buildRefinementContextMessages } from "@/lib/server/refinement-prompts";
import { noAccessMessage, resolveOpenRouterAccess } from "@/lib/server/openrouter-access";
import { isStorageNotFoundError, readData } from "@/lib/storage";
import {
  type RefinementMessageMetadata,
  type RefinementUIMessage,
  traceToModelMessages,
} from "@/lib/utils/trace-to-core";
import { isRecord } from "@/lib/utils/type-guards";
import { requireWorkspaceAccess } from "@/lib/workspace-access";

const TOOL_SERVER = getToolServerUrl();
const REFINE_MODEL = "anthropic/claude-sonnet-4";

async function loadTraceContext(
  workspaceId: string,
  stageId: string,
): Promise<{
  traceContext: ReturnType<typeof traceToModelMessages>;
  stageData: unknown | null;
}> {
  try {
    const raw = await readData(`${workspaceId}/run/${stageId}.json`);
    const stageData = JSON.parse(raw);

    if (stageData.llm_trace) {
      const baseTrace = stageData.llm_trace as LLMTrace;
      return {
        traceContext: traceToModelMessages(baseTrace.messages),
        stageData,
      };
    }

    return {
      traceContext: [],
      stageData,
    };
  } catch (e: unknown) {
    if (!isStorageNotFoundError(e)) {
      throw e;
    }
  }

  return {
    traceContext: [],
    stageData: null,
  };
}

function normalizeToolArgsForSchema(
  args: Record<string, unknown>,
  schema: unknown,
): Record<string, unknown> {
  if (!isRecord(schema) || !isRecord(schema.properties)) {
    return args;
  }

  const normalized: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(args)) {
    const propertySchema = schema.properties[key];
    if (isRecord(propertySchema) && propertySchema.type === "string" && typeof value !== "string") {
      normalized[key] = JSON.stringify(value);
      continue;
    }

    normalized[key] = value;
  }

  return normalized;
}

interface ToolInputExample {
  input: Record<string, unknown>;
}

function getStringEncodedJsonFields(schema: unknown): string[] {
  if (!isRecord(schema) || !isRecord(schema.properties)) {
    return [];
  }

  return Object.entries(schema.properties).flatMap(([key, value]) => {
    if (!isRecord(value) || value.type !== "string") {
      return [];
    }

    const description = typeof value.description === "string" ? value.description.toLowerCase() : "";
    return key.endsWith("_json") || description.includes("json string") || description.includes("json object")
      ? [key]
      : [];
  });
}

function buildToolDescription(
  stageId: string,
  toolName: string,
  description: string,
  schema: unknown,
): string {
  const notes: string[] = [];
  const jsonFields = getStringEncodedJsonFields(schema);

  if (jsonFields.length > 0) {
    notes.push(
      `Input requirement: ${jsonFields.map((field) => `\`${field}\``).join(", ")} must be valid JSON strings, not nested objects.`,
    );
  }

  if (stageId === "stage-4" && toolName === "validate_model") {
    notes.push(
      "Submit either `model_spec` or `priors` inside `model_json`. Do not mix both in the same call.",
    );
  }

  return notes.length > 0 ? `${description}\n\n${notes.join("\n")}` : description;
}

function getToolInputExamples(stageId: string, toolName: string): ToolInputExample[] | undefined {
  switch (`${stageId}/${toolName}`) {
    case "stage-1a/validate_latent_model":
      return [
        {
          input: {
            structure_json: JSON.stringify({
              constructs: [],
              edges: [],
            }),
          },
        },
      ];
    case "stage-1b/validate_measurement_model":
      return [
        {
          input: {
            measurement_json: JSON.stringify({
              model_clock: "1d",
              indicators: [],
            }),
          },
        },
      ];
    case "stage-4/search_literature":
      return [
        {
          input: {
            query: "daily stress sleep longitudinal effect size",
            parameter_name: "beta_stress_sleep",
          },
        },
      ];
    case "stage-4/validate_model":
      return [
        {
          input: {
            model_json: JSON.stringify({
              priors: {
                beta_stress_sleep: {
                  distribution: "Normal",
                  params: { mu: -0.2, sigma: 0.1 },
                },
              },
            }),
          },
        },
        {
          input: {
            model_json: JSON.stringify({
              model_spec: {
                likelihoods: [],
                parameters: [],
              },
            }),
          },
        },
      ];
    case "stage-6/get_model_info":
      return [
        {
          input: {
            sections: ["overview", "variables", "capabilities"],
            names: ["stress", "sleep_quality"],
          },
        },
      ];
    case "stage-6/simulate_intervention":
      return [
        {
          input: {
            action: { variable: "stress", mode: "shift", amount: -0.5 },
            outcome: "sleep_quality",
            query: { estimand: "trajectory", horizon_days: 30, projection: "latent" },
          },
        },
      ];
    case "stage-6/simulate_counterfactual":
      return [
        {
          input: {
            start: { time_index: 6 },
            action: { variable: "stress", mode: "shift", amount: -0.5 },
            outcome: "sleep_quality",
            query: { estimand: "trajectory", horizon_days: 30, projection: "latent" },
          },
        },
      ];
    default:
      return undefined;
  }
}

function logModelStepRequest(stageId: string, event: { stepNumber: number; request?: { body?: unknown }; warnings?: unknown }) {
  if (process.env.NODE_ENV === "test") {
    return;
  }

  console.info("[refine] model step request", {
    stageId,
    stepNumber: event.stepNumber,
    requestBody: event.request?.body ?? null,
    warnings: event.warnings ?? null,
  });
}

async function readToolErrorMessage(response: Response): Promise<string> {
  const bodyText = await response.text();
  if (!bodyText.trim()) {
    return `Tool execution failed with HTTP ${response.status}`;
  }

  try {
    const parsed = JSON.parse(bodyText) as unknown;
    if (isRecord(parsed)) {
      if (typeof parsed.error === "string" && parsed.error.trim()) {
        return parsed.error;
      }
      const detail = parsed.detail;
      if (typeof detail === "string" && detail.trim()) {
        return detail;
      }
      if (isRecord(detail) && typeof detail.message === "string" && detail.message.trim()) {
        return detail.message;
      }
    }
  } catch {
    // Fall back to the raw response text below.
  }

  return bodyText;
}

/**
 * POST /api/refine
 *
 * Streams a refinement conversation with full pipeline trace context
 * and the same tools the pipeline used (proxied to Python for execution).
 *
 * Body: { messages, workspaceId, stageId }
 */
export async function POST(req: Request) {
  const access = await resolveOpenRouterAccess();
  if (access.mode === "none") {
    return NextResponse.json({ error: noAccessMessage(access.reason) }, { status: 402 });
  }

  const { messages, workspaceId, stageId, pendingStagePatch } = await req.json();
  if (!Array.isArray(messages)) {
    return NextResponse.json({ error: "messages must be an array" }, { status: 400 });
  }

  const uiMessages = messages as RefinementUIMessage[];
  const hasWorkspaceId = typeof workspaceId === "string" && workspaceId.trim().length > 0;
  const safeStageId = typeof stageId === "string" ? stageId.trim() : "";
  const safePendingStagePatch = isRecord(pendingStagePatch) ? pendingStagePatch : {};

  if (stageId && (!safeStageId || /[\\/]/.test(safeStageId))) {
    return NextResponse.json({ error: "Invalid stageId format" }, { status: 400 });
  }
  const workspaceAccess = hasWorkspaceId ? await requireWorkspaceAccess(req, workspaceId) : null;
  if (workspaceAccess && !workspaceAccess.ok) {
    return workspaceAccess.response;
  }
  const normalizedWorkspaceId = workspaceAccess?.ok ? workspaceAccess.workspaceId : null;

  const { traceContext, stageData } =
    normalizedWorkspaceId && safeStageId
      ? await loadTraceContext(normalizedWorkspaceId, safeStageId)
      : { traceContext: [], stageData: null };
  const refinementContext = buildRefinementContextMessages(
    safeStageId,
    stageData,
    safePendingStagePatch,
  );

  // Build tools if this is an interactive stage
  const toolDefs =
    normalizedWorkspaceId && safeStageId && INTERACTIVE_STAGES.includes(safeStageId)
      ? STAGE_TOOLS[safeStageId] ?? []
      : [];

  const tools = Object.fromEntries(
    toolDefs.map((t) => [
      t.name,
      tool({
        description: buildToolDescription(safeStageId, t.name, t.description, t.parameters),
        inputSchema: jsonSchema<Record<string, unknown>>(t.parameters),
        ...(t.result ? { outputSchema: jsonSchema(t.result) } : {}),
        ...(getToolInputExamples(safeStageId, t.name)
          ? { inputExamples: getToolInputExamples(safeStageId, t.name) }
          : {}),
        strict: true,
        execute: async (args: Record<string, unknown>) => {
          if (!normalizedWorkspaceId) {
            throw new Error("Tool execution requires a workspace");
          }
          const normalizedArgs = normalizeToolArgsForSchema(args, t.parameters);
          const res = await fetch(
            `${TOOL_SERVER}/api/tools/${safeStageId}/${t.name}`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                workspace_id: normalizedWorkspaceId,
                input: normalizedArgs,
              }),
            },
          );
          if (!res.ok) {
            throw new Error(await readToolErrorMessage(res));
          }
          const data = await res.json();

          if (data.stage_output) {
            nextStagePatch = {
              ...nextStagePatch,
              ...data.stage_output,
            };
          }

          return data.result;
        },
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
      } as any),
    ]),
  );

  const modelMessages = await convertToModelMessages(uiMessages);
  const startedAt = Date.now();
  let nextStagePatch = { ...safePendingStagePatch };
  const openrouter = createOpenRouter({ apiKey: access.apiKey });
  const model = wrapLanguageModel({
    model: openrouter(REFINE_MODEL),
    middleware: addToolInputExamplesMiddleware({
      prefix: "Input Examples:",
    }),
  });

  const result = streamText({
    model,
    messages: [...traceContext, ...refinementContext, ...modelMessages],
    onStepFinish: (event) => {
      logModelStepRequest(safeStageId, event);
    },
    experimental_include: { requestBody: true },
    ...(Object.keys(tools).length > 0 ? { tools, stopWhen: stepCountIs(10) } : {}),
  });

  return result.toUIMessageStreamResponse({
    originalMessages: uiMessages,
    messageMetadata: ({ part }): RefinementMessageMetadata | undefined => {
      if (part.type !== "finish") {
        return undefined;
      }

      return {
        durationSeconds: (Date.now() - startedAt) / 1000,
        stagePatch: nextStagePatch,
        usage: {
          inputTokens: part.totalUsage.inputTokens ?? undefined,
          outputTokens: part.totalUsage.outputTokens ?? undefined,
          reasoningTokens:
            part.totalUsage.outputTokenDetails.reasoningTokens ??
            part.totalUsage.reasoningTokens ??
            undefined,
        },
      };
    },
  });
}
