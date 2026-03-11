"use client";

import { Badge } from "@/components/ui/badge";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import type { LLMTrace } from "@causal-ssm/api-types";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { CheckCircle, Clock, Cpu, Loader2, Play, Send } from "lucide-react";
import { type FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ChatMessages } from "./chat-messages";

function TraceSummary({ trace }: { trace: LLMTrace }) {
  const { usage } = trace;
  return (
    <div className="shrink-0 flex flex-wrap items-center gap-2 border-b bg-background pb-2 text-xs">
      <Badge variant="secondary" className="gap-1 text-[10px]">
        <Cpu className="h-3 w-3" />
        {trace.model}
      </Badge>
      <span className="text-muted-foreground">
        {formatCompact(usage.input_tokens)} in / {formatCompact(usage.output_tokens)} out
      </span>
      {usage.reasoning_tokens ? (
        <span className="text-muted-foreground">
          ({formatCompact(usage.reasoning_tokens)} reasoning)
        </span>
      ) : null}
      <span className="ml-auto flex items-center gap-1 text-muted-foreground">
        <Clock className="h-3 w-3" />
        {trace.total_time_seconds.toFixed(1)}s
      </span>
    </div>
  );
}

export function LLMTracePanel({
  trace,
  runId,
  stageId,
  interactive = true,
}: {
  trace: LLMTrace;
  runId?: string;
  stageId?: string;
  interactive?: boolean;
}) {
  const initialMessages = useMemo(() => traceToUIMessages(trace), [trace]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(false);

  const canRefine = interactive && !!runId && !!stageId;

  const transport = useMemo(
    () =>
      canRefine
        ? new DefaultChatTransport({ api: "/api/refine", body: { runId, stageId } })
        : undefined,
    [runId, stageId, canRefine],
  );

  const { messages, sendMessage, status } = useChat({
    transport: transport ?? new DefaultChatTransport({ api: "/api/refine" }),
    messages: initialMessages,
  });

  const displayMessages = canRefine ? messages : initialMessages;
  const isLoading = status === "streaming" || status === "submitted";
  const hasNewMessages = canRefine && messages.length > initialMessages.length;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [displayMessages.length]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading || !canRefine) return;
    setInput("");
    sendMessage({ text });
  }

  const handleApply = useCallback(async () => {
    if (applying || applied || !canRefine) return;
    setApplying(true);
    try {
      const apiMessages = messages.map((msg: UIMessage) => ({
        role: msg.role,
        content: msg.parts
          .filter((p) => p.type === "text")
          .map((p) => (p as { text: string }).text)
          .join("\n"),
      }));

      const res = await fetch("/api/refine/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: apiMessages, runId, stageId }),
      });

      if (!res.ok) {
        const error = await res.text();
        console.error("Apply failed:", error);
        return;
      }

      const result = await res.json();
      if (result.ok) {
        setApplied(true);
        if (result.flowRunId) {
          window.location.href = `/analysis/${result.flowRunId}`;
        }
      }
    } finally {
      setApplying(false);
    }
  }, [messages, runId, stageId, applying, applied, canRefine]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <TraceSummary trace={trace} />
      <div className="min-h-0 flex-1 overflow-y-auto">
        <ChatMessages messages={displayMessages} />
        <div ref={bottomRef} />
      </div>

      {canRefine && (
        <div className="shrink-0 flex flex-col gap-2">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Refine the output..."
              disabled={isLoading || applying}
              className="flex-1 rounded-md border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim() || applying}
              className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
            >
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </button>
          </form>

          {hasNewMessages && !isLoading && (
            <button
              type="button"
              onClick={handleApply}
              disabled={applying || applied}
              className="inline-flex w-full items-center justify-center gap-1.5 rounded-md bg-green-600 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-green-700 disabled:opacity-50"
            >
              {applying ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Applying...
                </>
              ) : applied ? (
                <>
                  <CheckCircle className="h-4 w-4" />
                  Applied — Re-running downstream
                </>
              ) : (
                <>
                  <Play className="h-4 w-4" />
                  Apply Changes & Re-run
                </>
              )}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
