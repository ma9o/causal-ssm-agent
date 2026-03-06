import { openrouter } from "@openrouter/ai-sdk-provider";
import { streamText } from "ai";

const MODEL = process.env.OPENROUTER_MODEL ?? "anthropic/claude-sonnet-4";

export async function POST(req: Request) {
  const { messages } = await req.json();

  const result = streamText({
    model: openrouter(MODEL),
    messages,
  });

  return result.toUIMessageStreamResponse();
}
