export interface NormalizedTraceToolCall {
  toolCallId: string;
  toolName: string;
  input: unknown;
}

export function normalizeTraceToolCall(
  toolCall: unknown,
): NormalizedTraceToolCall | null {
  if (typeof toolCall !== "object" || toolCall === null) {
    return null;
  }

  const record = toolCall as Record<string, unknown>;
  const toolCallId = typeof record.id === "string" && record.id.length > 0 ? record.id : null;
  if (!toolCallId) {
    return null;
  }

  const nestedFunction =
    typeof record.function === "object" && record.function !== null
      ? (record.function as Record<string, unknown>)
      : null;
  const toolName =
    typeof record.name === "string" && record.name.length > 0
      ? record.name
      : typeof nestedFunction?.name === "string" && nestedFunction.name.length > 0
        ? nestedFunction.name
        : null;
  if (!toolName) {
    return null;
  }

  const rawArguments = record.arguments ?? nestedFunction?.arguments;
  if (typeof rawArguments !== "string") {
    return {
      toolCallId,
      toolName,
      input: rawArguments ?? {},
    };
  }

  try {
    return {
      toolCallId,
      toolName,
      input: JSON.parse(rawArguments),
    };
  } catch (err) {
    console.warn("Failed to parse tool call arguments:", err);
    return {
      toolCallId,
      toolName,
      input: rawArguments,
    };
  }
}
