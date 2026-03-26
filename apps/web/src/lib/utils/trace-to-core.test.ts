import { modelMessageSchema } from "ai";
import { describe, expect, it } from "vitest";

import { traceToModelMessages } from "./trace-to-core";

describe("traceToModelMessages", () => {
  it("emits AI SDK-compatible tool call parts", () => {
    const messages = traceToModelMessages([
      {
        role: "assistant",
        content: "I validated the measurement model.",
        tool_calls: [
          {
            id: "call-1",
            name: "validate_measurement_model",
            arguments: JSON.stringify({
              candidate_json: '{"indicators":[]}',
            }),
          },
        ],
        tool_is_error: false,
      },
      {
        role: "tool",
        content: "VALID",
        tool_call_id: "call-1",
        tool_name: "validate_measurement_model",
        tool_result: "VALID",
        tool_is_error: false,
      },
    ]);

    expect(messages).toEqual([
      {
        role: "assistant",
        content: [
          { type: "text", text: "I validated the measurement model." },
          {
            type: "tool-call",
            toolCallId: "call-1",
            toolName: "validate_measurement_model",
            input: { candidate_json: '{"indicators":[]}' },
          },
        ],
      },
      {
        role: "tool",
        content: [
          {
            type: "tool-result",
            toolCallId: "call-1",
            toolName: "validate_measurement_model",
            output: { type: "text", value: "VALID" },
          },
        ],
      },
    ]);

    expect(modelMessageSchema.array().parse(messages)).toEqual(messages);
  });
});
