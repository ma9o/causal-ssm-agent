/**
 * Convert pipeline TraceMessage[] → AI SDK CoreMessage[] (lossless, server-side).
 *
 * Used by the refinement route to prepend the full pipeline conversation
 * as LLM context. Unlike traceToUIMessages (display-only), this preserves
 * all information: tool calls, reasoning, system messages.
 */
import type { TraceMessage } from "@causal-ssm/api-types";
import type { CoreMessage } from "ai";

export function traceToCoreMessages(messages: TraceMessage[]): CoreMessage[] {
  const result: CoreMessage[] = [];

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
              result: msg.tool_result ?? msg.content,
              isError: msg.tool_is_error,
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
            args: unknown;
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
          content.push({
            type: "tool-call",
            toolCallId: tc.id,
            toolName: tc.name,
            args:
              typeof tc.arguments === "string"
                ? JSON.parse(tc.arguments)
                : tc.arguments,
          });
        }
      }

      result.push({
        role: "assistant",
        content: content.length > 0 ? (content as CoreMessage["content"]) : msg.content,
      });
      continue;
    }
  }

  return result;
}
