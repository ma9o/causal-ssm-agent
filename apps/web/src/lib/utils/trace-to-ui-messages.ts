/**
 * Convert pipeline LLMTrace → Vercel AI SDK UIMessage[].
 *
 * The pipeline stores a flat list of TraceMessages (system, user, assistant, tool).
 * The AI SDK uses a parts-based UIMessage format where:
 * - Each assistant message contains text, reasoning, and tool-call parts
 * - Tool results are merged into the assistant message's tool parts (state: "output-available")
 * - System/user messages become simple text-part messages
 */
import type { LLMTrace, TraceMessage } from "@causal-ssm/api-types";
import type { UIMessage } from "ai";

type Part = UIMessage["parts"][number];

/**
 * Build a UIMessage from role + parts, generating a stable id from the index.
 */
function makeMsg(
  role: "system" | "user" | "assistant",
  parts: Part[],
  idx: number,
): UIMessage {
  return { id: `trace-${idx}`, role, parts };
}

/**
 * Convert a pipeline LLMTrace to an array of AI SDK UIMessages.
 *
 * Strategy:
 * - system/user → one UIMessage with a single text part
 * - assistant → one UIMessage with text + reasoning + dynamic-tool parts
 * - tool messages following an assistant are folded into the preceding
 *   assistant's tool parts as output-available / output-error
 */
export function traceToUIMessages(trace: LLMTrace): UIMessage[] {
  const messages: UIMessage[] = [];
  const traceMessages = trace.messages;

  let i = 0;
  while (i < traceMessages.length) {
    const msg = traceMessages[i];

    if (msg.role === "system" || msg.role === "user") {
      messages.push(makeMsg(msg.role, [{ type: "text", text: msg.content }], i));
      i++;
      continue;
    }

    if (msg.role === "assistant") {
      const parts: Part[] = [];

      // Reasoning
      if (msg.reasoning) {
        parts.push({ type: "reasoning", text: msg.reasoning, providerMetadata: undefined });
      }

      // Text content
      if (msg.content) {
        parts.push({ type: "text", text: msg.content });
      }

      // Tool calls — start as input-available, then look ahead for matching tool results
      const toolParts: Map<string, Part> = new Map();
      if (msg.tool_calls) {
        for (const tc of msg.tool_calls) {
          const part: Part = {
            type: "dynamic-tool" as const,
            toolCallId: tc.id,
            toolName: tc.name,
            state: "input-available" as const,
            input: tc.arguments,
          };
          toolParts.set(tc.id, part);
          parts.push(part);
        }
      }

      // Consume following tool messages and merge them into the tool parts
      let j = i + 1;
      while (j < traceMessages.length && traceMessages[j].role === "tool") {
        const toolMsg = traceMessages[j];
        const callId = toolMsg.tool_call_id;
        if (callId && toolParts.has(callId)) {
          // Find the part in the parts array and replace it
          const idx = parts.findIndex(
            (p) => p.type === "dynamic-tool" && "toolCallId" in p && p.toolCallId === callId,
          );
          if (idx !== -1) {
            if (toolMsg.tool_is_error) {
              parts[idx] = {
                type: "dynamic-tool" as const,
                toolCallId: callId,
                toolName: toolMsg.tool_name ?? "unknown",
                state: "output-error" as const,
                input: (parts[idx] as { input: unknown }).input,
                errorText: toolMsg.tool_result ?? toolMsg.content,
              };
            } else {
              parts[idx] = {
                type: "dynamic-tool" as const,
                toolCallId: callId,
                toolName: toolMsg.tool_name ?? "unknown",
                state: "output-available" as const,
                input: (parts[idx] as { input: unknown }).input,
                output: toolMsg.tool_result ?? toolMsg.content,
              };
            }
          }
        }
        j++;
      }

      messages.push(makeMsg("assistant", parts, i));
      i = j; // skip past consumed tool messages
      continue;
    }

    // Orphan tool message (shouldn't happen, but handle gracefully)
    if (msg.role === "tool") {
      const toolCallId = msg.tool_call_id ?? `orphan-${i}`;
      const part: Part = msg.tool_is_error
        ? {
            type: "dynamic-tool" as const,
            toolCallId,
            toolName: msg.tool_name ?? "unknown",
            state: "output-error" as const,
            input: {},
            errorText: msg.tool_result ?? msg.content,
          }
        : {
            type: "dynamic-tool" as const,
            toolCallId,
            toolName: msg.tool_name ?? "unknown",
            state: "output-available" as const,
            input: {},
            output: msg.tool_result ?? msg.content,
          };
      messages.push(makeMsg("assistant", [part], i));
      i++;
      continue;
    }

    // Unknown role — skip
    i++;
  }

  return messages;
}
