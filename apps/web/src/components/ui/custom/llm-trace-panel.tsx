"use client";

import { Badge } from "@/components/ui/badge";
import { formatCompact } from "@/lib/utils/format";
import { traceToUIMessages } from "@/lib/utils/trace-to-ui-messages";
import type { LLMTrace } from "@causal-ssm/api-types";
import { Clock, Cpu } from "lucide-react";
import { useMemo } from "react";
import { ChatMessages } from "./chat-messages";

function TraceSummary({ trace }: { trace: LLMTrace }) {
  const { usage } = trace;
  return (
    <div className="sticky top-0 z-10 flex flex-wrap items-center gap-2 border-b bg-background/95 pb-2 text-xs backdrop-blur">
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

export function LLMTracePanel({ trace }: { trace: LLMTrace }) {
  const uiMessages = useMemo(() => traceToUIMessages(trace), [trace]);

  return (
    <div className="flex flex-col gap-2">
      <TraceSummary trace={trace} />
      <ChatMessages messages={uiMessages} />
    </div>
  );
}
