/**
 * Convert pipeline TraceMessage[] → AI SDK ModelMessage[] (lossless, server-side).
 *
 * Used by the refinement route to prepend the full pipeline conversation
 * as LLM context. Unlike traceToUIMessages (display-only), this preserves
 * all information: tool calls, reasoning, system messages.
 */
import type { LLMTrace, TraceMessage } from "@nof1-causal-lab/api-types";
import type { AssistantModelMessage, ModelMessage, UIMessage } from "ai";
import { normalizeTraceToolCall } from "./trace-tool-call";

export interface RefinementUsage {
  inputTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
}

export interface RefinementMessageMetadata {
  durationSeconds?: number;
  stagePatch?: Record<string, unknown>;
  usage?: RefinementUsage;
}

export interface SuggestionAction {
  tool: string;
  input: Record<string, unknown>;
}

export interface SuggestionChip {
  label: string;
  action: SuggestionAction;
}

export interface RefinementDataTypes {
  suggestions: { suggestions: SuggestionChip[] };
  [key: string]: unknown;
}

export type RefinementUIMessage = UIMessage<RefinementMessageMetadata, RefinementDataTypes>;

export type SuggestionsDataPart = {
  type: "data-suggestions";
  id?: string;
  data: RefinementDataTypes["suggestions"];
};

export function isSuggestionsDataPart(part: unknown): part is SuggestionsDataPart {
  if (typeof part !== "object" || part === null) return false;
  const candidate = part as { type?: unknown; data?: unknown };
  if (candidate.type !== "data-suggestions") return false;
  if (typeof candidate.data !== "object" || candidate.data === null) return false;
  const list = (candidate.data as { suggestions?: unknown }).suggestions;
  if (!Array.isArray(list)) return false;
  return list.every((chip) => {
    if (typeof chip !== "object" || chip === null) return false;
    const c = chip as { label?: unknown; action?: unknown };
    if (typeof c.label !== "string") return false;
    if (typeof c.action !== "object" || c.action === null) return false;
    const a = c.action as { tool?: unknown; input?: unknown };
    return (
      typeof a.tool === "string" &&
      typeof a.input === "object" &&
      a.input !== null &&
      !Array.isArray(a.input)
    );
  });
}

function stringifyTraceValue(value: unknown): string {
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

export function traceToModelMessages(messages: TraceMessage[]): ModelMessage[] {
  const result: ModelMessage[] = [];

  for (const msg of messages) {
    if (msg.role === "system") {
      result.push({ role: "system", content: msg.content });
      continue;
    }

    if (msg.role === "user") {
      result.push({ role: "user", content: msg.content });
      continue;
    }

    if (msg.role === "tool") {
      if (msg.tool_call_id) {
        result.push({
          role: "tool",
          content: [
            {
              type: "tool-result",
              toolCallId: msg.tool_call_id,
              toolName: msg.tool_name ?? "unknown",
              output: typeof (msg.tool_result ?? msg.content) === "string"
                ? { type: "text" as const, value: msg.tool_result ?? msg.content }
                : { type: "json" as const, value: msg.tool_result ?? msg.content },
            },
          ],
        });
      }
      continue;
    }

    if (msg.role === "assistant") {
      const content: Array<
        | { type: "reasoning"; text: string }
        | { type: "text"; text: string }
        | {
            type: "tool-call";
            toolCallId: string;
            toolName: string;
            input: unknown;
          }
      > = [];

      if (msg.reasoning) {
        content.push({ type: "reasoning", text: msg.reasoning });
      }
      if (msg.content) {
        content.push({ type: "text", text: msg.content });
      }
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          const normalized = normalizeTraceToolCall(tc);
          if (!normalized) {
            continue;
          }

          content.push({
            type: "tool-call",
            toolCallId: normalized.toolCallId,
            toolName: normalized.toolName,
            input: normalized.input,
          });
        }
      }

      result.push({
        role: "assistant",
        content: content.length > 0 ? (content as AssistantModelMessage["content"]) : msg.content,
      });
      continue;
    }
  }

  return result;
}

export function uiMessagesToTraceMessages(messages: UIMessage[]): TraceMessage[] {
  const traceMessages: TraceMessage[] = [];

  for (const message of messages) {
    const text = message.parts
      .filter(
        (part): part is Extract<UIMessage["parts"][number], { type: "text" }> => part.type === "text",
      )
      .map((part) => part.text)
      .join("\n\n");

    if (message.role === "system" || message.role === "user") {
      traceMessages.push({
        role: message.role,
        content: text,
        tool_is_error: false,
      });
      continue;
    }

    if (message.role !== "assistant") {
      continue;
    }

    const reasoning = message.parts
      .filter(
        (part): part is Extract<UIMessage["parts"][number], { type: "reasoning" }> =>
          part.type === "reasoning",
      )
      .map((part) => part.text)
      .join("\n\n");

    const toolParts = message.parts.filter(
      (part): part is Extract<UIMessage["parts"][number], { type: "dynamic-tool" }> =>
        part.type === "dynamic-tool",
    );

    traceMessages.push({
      role: "assistant",
      content: text,
      ...(reasoning ? { reasoning } : {}),
      ...(toolParts.length > 0
        ? {
            tool_calls: toolParts.map((part) => ({
              id: part.toolCallId,
              name: part.toolName,
              arguments: part.input ?? {},
            })),
          }
        : {}),
      tool_is_error: false,
    });

    for (const part of toolParts) {
      if (part.state !== "output-available" && part.state !== "output-error") {
        continue;
      }

      traceMessages.push({
        role: "tool",
        content:
          part.state === "output-error" ? part.errorText : stringifyTraceValue(part.output),
        tool_call_id: part.toolCallId,
        tool_name: part.toolName,
        tool_result:
          part.state === "output-error" ? part.errorText : stringifyTraceValue(part.output),
        tool_is_error: part.state === "output-error",
      });
    }
  }

  return traceMessages;
}

export function summarizeRefinementMessages(messages: RefinementUIMessage[]): {
  durationSeconds: number;
  stagePatch: Record<string, unknown>;
  usage: RefinementUsage | null;
} {
  let durationSeconds = 0;
  let inputTokens = 0;
  let outputTokens = 0;
  let reasoningTokens = 0;
  let hasUsage = false;
  const stagePatch: Record<string, unknown> = {};

  for (const message of messages) {
    const metadata = message.metadata;
    if (!metadata) {
      continue;
    }

    durationSeconds += metadata.durationSeconds ?? 0;

    if (metadata.stagePatch) {
      Object.assign(stagePatch, metadata.stagePatch);
    }

    if (metadata.usage) {
      hasUsage = true;
      inputTokens += metadata.usage.inputTokens ?? 0;
      outputTokens += metadata.usage.outputTokens ?? 0;
      reasoningTokens += metadata.usage.reasoningTokens ?? 0;
    }
  }

  return {
    durationSeconds,
    stagePatch,
    usage: hasUsage
      ? {
          inputTokens,
          outputTokens,
          ...(reasoningTokens > 0 ? { reasoningTokens } : {}),
        }
      : null,
  };
}

export function mergePersistedTrace(
  baseTrace: LLMTrace | null,
  messages: UIMessage[],
  {
    durationSeconds,
    usage,
  }: {
    durationSeconds: number;
    usage: RefinementUsage | null;
  },
): LLMTrace {
  const baseUsage = baseTrace?.usage;
  const reasoningTokens =
    (baseUsage?.reasoning_tokens ?? 0) + (usage?.reasoningTokens ?? 0);

  return {
    messages: [...(baseTrace?.messages ?? []), ...uiMessagesToTraceMessages(messages)],
    model: baseTrace?.model ?? "openrouter/anthropic/claude-sonnet-4",
    total_time_seconds: (baseTrace?.total_time_seconds ?? 0) + durationSeconds,
    usage: {
      input_tokens: (baseUsage?.input_tokens ?? 0) + (usage?.inputTokens ?? 0),
      output_tokens: (baseUsage?.output_tokens ?? 0) + (usage?.outputTokens ?? 0),
      ...(reasoningTokens > 0 ? { reasoning_tokens: reasoningTokens } : {}),
    },
  };
}
