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

  it("accepts persisted nested function tool call shapes", () => {
    const messages = traceToModelMessages([
      {
        role: "assistant",
        content: "I repaired the measurement model.",
        tool_calls: [
          {
            id: "call-2",
            type: "function",
            function: {
              name: "validate_measurement_model",
              arguments: JSON.stringify({
                measurement_json: '{"model_clock":"1d","indicators":[]}',
              }),
            },
          },
        ],
        tool_is_error: false,
      },
      {
        role: "tool",
        content: "VALID",
        tool_call_id: "call-2",
        tool_name: "validate_measurement_model",
        tool_result: "VALID",
        tool_is_error: false,
      },
    ]);

    expect(modelMessageSchema.array().parse(messages)).toEqual(messages);
    expect(messages[0]).toEqual({
      role: "assistant",
      content: [
        { type: "text", text: "I repaired the measurement model." },
        {
          type: "tool-call",
          toolCallId: "call-2",
          toolName: "validate_measurement_model",
          input: {
            measurement_json: '{"model_clock":"1d","indicators":[]}',
          },
        },
      ],
    });
  });
});
