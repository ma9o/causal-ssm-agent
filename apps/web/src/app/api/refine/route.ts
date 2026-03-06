import { openrouter } from "@openrouter/ai-sdk-provider";
import { streamText } from "ai";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openrouter("anthropic/claude-sonnet-4"),
    messages,
  });

  return result.toUIMessageStreamResponse();
}
