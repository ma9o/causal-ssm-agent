"use client";

import { Badge } from "@/components/ui/badge";
import { getUserApiKey } from "@/lib/auth";
import {
  refinementNeedsActivation,
  useRefinement,
} from "@/lib/contexts/refinement-context";
import type { RefinementUIMessage } from "@/lib/utils/trace-to-core";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import { useChat } from "@ai-sdk/react";
import type { LLMTrace, StageId } from "@causal-ssm/api-types";
import { INTERACTIVE_STAGES } from "@causal-ssm/api-types";
import { DefaultChatTransport } from "ai";
import { Clock, Cpu, Loader2, MessageSquare, Send } from "lucide-react";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ChatMessages } from "./chat-messages";

const EMPTY_STAGE_PATCH: Record<string, unknown> = {};
const EMPTY_REFINEMENT_MESSAGES: RefinementUIMessage[] = [];

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

  const normalizedStageId = stageId as StageId | undefined;
  const {
    refiningStageId,
    pendingStagePatches,
    refinementMessages: savedRefinementMessages,
    requestRefinement,
    markSettled,
    setPendingMaterialization,
  } = useRefinement();

  const canRefine = interactive && !!workspaceId && !!stageId && INTERACTIVE_STAGES.includes(stageId);
  const pendingStagePatch = normalizedStageId
    ? (pendingStagePatches[normalizedStageId] ?? EMPTY_STAGE_PATCH)
    : EMPTY_STAGE_PATCH;
  const initialRefinementMessages = normalizedStageId
    ? (savedRefinementMessages[normalizedStageId] ?? EMPTY_REFINEMENT_MESSAGES)
    : EMPTY_REFINEMENT_MESSAGES;

  // Refinement chat — independent from trace, NOT initialized with trace messages.
  // The server prepends the trace as CoreMessages for LLM context.
  const transport = useMemo(() => {
    if (!canRefine) return undefined;
    const apiKey = getUserApiKey();
    return new DefaultChatTransport<RefinementUIMessage>({
      api: "/api/refine",
      body: { workspaceId, stageId, pendingStagePatch },
      ...(apiKey ? { headers: { "x-openrouter-key": apiKey } } : {}),
    });
  }, [workspaceId, stageId, canRefine, pendingStagePatch]);

  const {
    messages: refinementMessages,
    sendMessage,
    status,
  } = useChat<RefinementUIMessage>({
    messages: initialRefinementMessages,
    transport: transport ?? new DefaultChatTransport<RefinementUIMessage>({ api: "/api/refine" }),
    onFinish: ({ message, messages }) => {
      if (!normalizedStageId) {
        return;
      }

      setPendingMaterialization(normalizedStageId, {
        messages,
        stagePatch: message.metadata?.stagePatch ?? pendingStagePatch,
      });
    },
  });

  const isLoading = status === "streaming" || status === "submitted";
  const hasRefinement = refinementMessages.length > 0;

  useEffect(() => {
    if (!normalizedStageId || refinementMessages.length === 0) {
      return;
    }

    setPendingMaterialization(normalizedStageId, {
      messages: refinementMessages,
    });
  }, [normalizedStageId, refinementMessages, setPendingMaterialization]);

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

    // First message enters refinement mode. Non-terminal stages open the invalidation modal.
    if (refinementNeedsActivation(stageId as StageId, refiningStageId)) {
      queuedMessageRef.current = text;
      requestRefinement(stageId as StageId);
      return;
    }

    // Stage is already in refinement mode, or it's terminal and can refine in place.
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
            <span>Follow-up Chat</span>
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
            placeholder="Ask a follow-up question or request a change..."
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
