"use client";

import { Badge } from "@/components/ui/badge";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import type { LLMTrace } from "@nof1-causal-lab/api-types";
import { Clock, Cpu } from "lucide-react";
import { useEffect, useMemo, useRef } from "react";
import { ChatMessages, type SimulationResult } from "./chat-messages";

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
 * Pure presentational component for the read-only LLM trace panel.
 * Safe for Storybook — no app-state dependency.
 */
export function LLMTracePanelView({
  trace,
  selectedSimulationKey,
  onSelectSimulation,
}: {
  trace: LLMTrace;
  selectedSimulationKey?: string;
  onSelectSimulation?: (key: string, result: SimulationResult) => void;
}) {
  const traceMessages = useMemo(() => traceToUIMessages(trace), [trace]);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    // Defer until after parent layout/animation settles so the container
    // has its final constrained height (e.g. OutputWithTrace motion).
    const raf = requestAnimationFrame(() => {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    });
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <TraceSummary trace={trace} />
      <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
        <ChatMessages
          messages={traceMessages}
          selectedSimulationKey={selectedSimulationKey}
          onSelectSimulation={onSelectSimulation}
        />
      </div>
    </div>
  );
}
