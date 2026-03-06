"use client";

import { ChatMessages } from "@/components/ui/custom/chat-messages";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import type { LLMTrace } from "@causal-ssm/api-types";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { Bot, Loader2, Send } from "lucide-react";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

export function RefinementPanel({
  trace,
  runId,
  stageId,
  onClose,
}: {
  trace: LLMTrace;
  runId: string;
  stageId: string;
  onClose: () => void;
}) {
  const initialMessages = useMemo(() => traceToUIMessages(trace), [trace]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");

  const transport = useMemo(
    () => new DefaultChatTransport({ api: "/api/refine", body: { runId, stageId } }),
    [runId, stageId],
  );

  const { messages, sendMessage, status } = useChat({
    transport,
    messages: initialMessages,
  });

  const isLoading = status === "streaming" || status === "submitted";

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading) return;
    setInput("");
    sendMessage({ text });
  }

  return (
    <div className="flex h-full flex-col">
      <button
        type="button"
        onClick={onClose}
        className="inline-flex w-full shrink-0 items-center justify-center gap-1.5 rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition-colors"
      >
        <Bot className="h-3.5 w-3.5" />
        Hide Refinement
      </button>

      <div className="min-h-0 flex-1 overflow-y-auto rounded-lg border bg-muted/30 p-3 mt-3">
        <ChatMessages messages={messages} />
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="mt-3 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Refine the output..."
          disabled={isLoading}
          className="flex-1 rounded-md border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim()}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
        >
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </button>
      </form>
    </div>
  );
}
