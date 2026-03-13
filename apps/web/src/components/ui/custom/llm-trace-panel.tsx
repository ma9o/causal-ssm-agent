"use client";

import { Badge } from "@/components/ui/badge";
import { getUserApiKey } from "@/lib/auth";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import type { LLMTrace } from "@causal-ssm/api-types";
import { INTERACTIVE_STAGES } from "@causal-ssm/api-types";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { CheckCircle, Clock, Cpu, Loader2, MessageSquare, Play, Send } from "lucide-react";
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

/**
 * LLM Trace Panel — three modes:
 *
 * 1. Read-only (no userId/stageId or non-refinable stage): just shows the trace
 * 2. Completed + refinable: trace (read-only) + refinement chat (interactive)
 * 3. Refining: trace + ongoing refinement conversation with tools
 *
 * Key design: the trace is display-only (top section). Refinement is a
 * separate chat (bottom section). The server prepends trace as LLM context
 * via CoreMessages, so the refinement chat starts empty on the client.
 */
export function LLMTracePanel({
  trace,
  userId,
  stageId,
  interactive = true,
}: {
  trace: LLMTrace;
  userId?: string;
  stageId?: string;
  interactive?: boolean;
}) {
  const traceMessages = useMemo(() => traceToUIMessages(trace), [trace]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");
  const [applying, setApplying] = useState(false);
  const [applied, setApplied] = useState(false);

  const canRefine =
    interactive && !!userId && !!stageId && INTERACTIVE_STAGES.includes(stageId);

  // Refinement chat — independent from trace, NOT initialized with trace messages.
  // The server prepends the trace as CoreMessages for LLM context.
  const transport = useMemo(() => {
    if (!canRefine) return undefined;
    const apiKey = getUserApiKey();
    return new DefaultChatTransport({
      api: "/api/refine",
      body: { userId, stageId },
      ...(apiKey ? { headers: { "x-openrouter-key": apiKey } } : {}),
    });
  }, [userId, stageId, canRefine]);

  const {
    messages: refinementMessages,
    sendMessage,
    status,
  } = useChat({
    transport: transport ?? new DefaultChatTransport({ api: "/api/refine" }),
  });

  const isLoading = status === "streaming" || status === "submitted";
  const hasRefinement = refinementMessages.length > 0;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [traceMessages.length, refinementMessages.length]);

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
      const apiMessages = refinementMessages.map((msg) => ({
        role: msg.role,
        content: msg.parts
          .filter((p) => p.type === "text")
          .map((p) => (p as { text: string }).text)
          .join("\n"),
      }));

      const res = await fetch("/api/refine/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: apiMessages, userId, stageId }),
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
          window.location.href = `/analysis/${userId}?flowRunId=${result.flowRunId}`;
        }
      }
    } finally {
      setApplying(false);
    }
  }, [refinementMessages, userId, stageId, applying, applied, canRefine]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <TraceSummary trace={trace} />

      {/* Trace messages — read-only */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        <ChatMessages messages={traceMessages} />

        {/* Separator between trace and refinement */}
        {hasRefinement && (
          <div className="my-3 flex items-center gap-2 text-xs text-muted-foreground">
            <div className="flex-1 border-t" />
            <MessageSquare className="h-3 w-3" />
            <span>Refinement</span>
            <div className="flex-1 border-t" />
          </div>
        )}

        {/* Refinement messages — interactive */}
        {hasRefinement && <ChatMessages messages={refinementMessages} />}

        <div ref={bottomRef} />
      </div>

      {/* Refinement input */}
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

          {/* Apply button — visible after refinement, when not streaming */}
          {hasRefinement && !isLoading && (
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
