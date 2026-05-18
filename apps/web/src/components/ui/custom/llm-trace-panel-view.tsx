"use client";

import { Badge } from "@/components/ui/badge";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import type { LLMTrace } from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { Clock, Cpu, Loader2, MessageSquare, Send } from "lucide-react";
import { type FormEvent, useCallback, useEffect, useMemo, useRef } from "react";
import { ChatMessages } from "./chat-messages";

function TraceSummary({ trace }: { trace: LLMTrace }) {
  const { usage } = trace;
  return (
    <div className="shrink-0 flex flex-wrap items-center gap-2 border-b bg-background pb-2 text-xs">
      <Badge variant="secondary" className="gap-1 text-[11px]">
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
 * Pure presentational component for the LLM trace panel.
 * No dependency on AI SDK hooks or refinement context — safe for Storybook.
 */
export function LLMTracePanelView({
  trace,
  refinementMessages = [],
  canRefine = false,
  isLoading = false,
  input = "",
  onInputChange,
  onSubmit,
}: {
  trace: LLMTrace;
  refinementMessages?: UIMessage[];
  canRefine?: boolean;
  isLoading?: boolean;
  input?: string;
  onInputChange?: (value: string) => void;
  onSubmit?: (e: FormEvent) => void;
}) {
  const traceMessages = useMemo(() => traceToUIMessages(trace), [trace]);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const hasRefinement = refinementMessages.length > 0;

  const resizeTextarea = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, []);

  const messageCount = traceMessages.length + refinementMessages.length;
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Defer until after parent layout/animation settles so the container
    // has its final constrained height (e.g. StageWithTrace motion).
    const raf = requestAnimationFrame(() => {
      el.scrollTo({
        top: el.scrollHeight,
        behavior: isLoading ? "auto" : "smooth",
      });
    });
    return () => cancelAnimationFrame(raf);
  }, [messageCount, refinementMessages, isLoading]);

  // Auto-focus the refinement input on mount
  useEffect(() => {
    if (!canRefine) return;
    const raf = requestAnimationFrame(() => inputRef.current?.focus());
    return () => cancelAnimationFrame(raf);
  }, [canRefine]);

  // Resize textarea on external input changes:
  // - Shrink back when cleared (e.g. after submit)
  // - Auto-expand prefilled text after layout settles so "Propose Fixes"
  //   prompts open at the expected capped height
  useEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    if (!input) {
      el.style.height = "auto";
      return;
    }

    const raf = requestAnimationFrame(() => {
      resizeTextarea();
    });
    return () => cancelAnimationFrame(raf);
  }, [input, resizeTextarea]);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <TraceSummary trace={trace} />

      {/* Trace messages — read-only */}
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
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
        {hasRefinement && <ChatMessages messages={refinementMessages} streaming={isLoading} />}
      </div>

      {/* Refinement input */}
      {canRefine && (
        <form onSubmit={onSubmit} className="shrink-0 flex items-end gap-2">
          <textarea
            ref={inputRef}
            value={input}
            rows={1}
            onChange={(e) => {
              onInputChange?.(e.target.value);
              resizeTextarea();
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                onSubmit?.(e as unknown as FormEvent);
              }
            }}
            placeholder="Ask a follow-up question or request a change..."
            disabled={isLoading}
            className="flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm leading-5 placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-50"
            style={{ maxHeight: "calc(5 * 1.25rem + 1rem)" }}
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
