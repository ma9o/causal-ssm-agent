"use client";

import { Badge } from "@/components/ui/badge";
import { getUserApiKey } from "@/lib/auth";
import { useRefinement } from "@/lib/contexts/refinement-context";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { useChat } from "@ai-sdk/react";
import type { LLMTrace, StageId } from "@causal-ssm/api-types";
import { INTERACTIVE_STAGES } from "@causal-ssm/api-types";
import { DefaultChatTransport } from "ai";
import { Clock, Cpu, Loader2, MessageSquare, Send } from "lucide-react";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
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
 * 1. Read-only (no workspaceId/stageId or non-refinable stage): just shows the trace
 * 2. Completed + refinable: trace (read-only) + refinement chat (interactive)
 * 3. Refining: trace + ongoing refinement conversation with tools
 *
 * On the first refinement message, a confirmation modal is shown (via
 * RefinementContext) warning that downstream stages will be invalidated.
 * The "Apply & Re-run" action has moved to the bottom-of-feed ResumeButton.
 */
export function LLMTracePanel({
  trace,
  workspaceId,
  stageId,
  interactive = true,
}: {
  trace: LLMTrace;
  workspaceId?: string;
  stageId?: string;
  interactive?: boolean;
}) {
  const traceMessages = useMemo(() => traceToUIMessages(trace), [trace]);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");
  const queuedMessageRef = useRef<string | null>(null);

  const { refiningStageId, invalidatedAfter, requestRefinement, markSettled } = useRefinement();

  const canRefine = interactive && !!workspaceId && !!stageId && INTERACTIVE_STAGES.includes(stageId);

  // Refinement chat — independent from trace, NOT initialized with trace messages.
  // The server prepends the trace as CoreMessages for LLM context.
  const transport = useMemo(() => {
    if (!canRefine) return undefined;
    const apiKey = getUserApiKey();
    return new DefaultChatTransport({
      api: "/api/refine",
      body: { workspaceId, stageId },
      ...(apiKey ? { headers: { "x-openrouter-key": apiKey } } : {}),
    });
  }, [workspaceId, stageId, canRefine]);

  const {
    messages: refinementMessages,
    sendMessage,
    status,
  } = useChat({
    transport: transport ?? new DefaultChatTransport({ api: "/api/refine" }),
  });

  const isLoading = status === "streaming" || status === "submitted";
  const hasRefinement = refinementMessages.length > 0;

  // Report settled state to the context so the ResumeButton can appear
  useEffect(() => {
    if (canRefine && refiningStageId === stageId) {
      markSettled(hasRefinement && !isLoading);
    }
  }, [canRefine, refiningStageId, stageId, hasRefinement, isLoading, markSettled]);

  // Send queued message after refinement is confirmed via the modal
  useEffect(() => {
    if (refiningStageId === stageId && queuedMessageRef.current) {
      const text = queuedMessageRef.current;
      queuedMessageRef.current = null;
      sendMessage({ text });
    }
  }, [refiningStageId, stageId, sendMessage]);

  const messageCount = traceMessages.length + refinementMessages.length;
  // biome-ignore lint/correctness/useExhaustiveDependencies: scroll on message count change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messageCount]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading || !canRefine) return;
    setInput("");

    // If downstream stages aren't invalidated yet, show the confirmation modal
    if (!invalidatedAfter) {
      queuedMessageRef.current = text;
      requestRefinement(stageId as StageId);
      return;
    }

    // Already confirmed — send immediately
    sendMessage({ text });
  }

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
        <form onSubmit={handleSubmit} className="shrink-0 flex gap-2">
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
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </form>
      )}
    </div>
  );
}
