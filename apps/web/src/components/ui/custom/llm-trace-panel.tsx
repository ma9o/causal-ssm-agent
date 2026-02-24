"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils/cn";
import type { LLMTrace, TraceMessage } from "@causal-ssm/api-types";
import { Bot, ChevronRight, Clock, Cpu, Wrench } from "lucide-react";

const compactNumber = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});

function TraceSummary({ trace }: { trace: LLMTrace }) {
  const { usage } = trace;
  return (
    <div className="sticky top-0 z-10 flex flex-wrap items-center gap-2 border-b bg-background/95 pb-2 text-xs backdrop-blur">
      <Badge variant="secondary" className="gap-1 text-[10px]">
        <Cpu className="h-3 w-3" />
        {trace.model}
      </Badge>
      <span className="text-muted-foreground">
        {compactNumber.format(usage.input_tokens)} in / {compactNumber.format(usage.output_tokens)} out
      </span>
      {usage.reasoning_tokens ? (
        <span className="text-muted-foreground">
          ({compactNumber.format(usage.reasoning_tokens)} reasoning)
        </span>
      ) : null}
      <span className="ml-auto flex items-center gap-1 text-muted-foreground">
        <Clock className="h-3 w-3" />
        {trace.total_time_seconds.toFixed(1)}s
      </span>
    </div>
  );
}

function SystemMessage({ msg }: { msg: TraceMessage }) {
  return (
    <Accordion>
      <AccordionItem value="system" className="border-l-2 border-muted-foreground/30 !border-b-0">
        <AccordionTrigger className="py-2 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <ChevronRight className="h-3 w-3" />
            System prompt
          </span>
        </AccordionTrigger>
        <AccordionContent>
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-[11px]">
            {msg.content}
          </pre>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function UserMessage({ msg }: { msg: TraceMessage }) {
  return (
    <div className="rounded-md border bg-background p-2.5">
      <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        User
      </div>
      <div className="whitespace-pre-wrap text-xs">{msg.content}</div>
    </div>
  );
}

function AssistantMessage({ msg, idx }: { msg: TraceMessage; idx: number }) {
  return (
    <div className="rounded-md border border-primary/20 bg-primary/5 p-2.5">
      <div className="mb-1 flex items-center gap-1.5">
        <Bot className="h-3 w-3 text-primary" />
        <span className="text-[10px] font-medium uppercase tracking-wide text-primary">
          Assistant
        </span>
      </div>

      {msg.reasoning && (
        <Accordion>
          <AccordionItem value={`reasoning-${idx}`} className="border-l-2 border-amber-400/50 !border-b-0">
            <AccordionTrigger
              className="py-1.5 text-[11px] text-amber-600 dark:text-amber-400"
            >
              Thinking
            </AccordionTrigger>
            <AccordionContent>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-amber-50/50 p-2 text-[11px] dark:bg-amber-950/20">
                {msg.reasoning}
              </pre>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {msg.content && <div className="mt-1 whitespace-pre-wrap text-xs">{msg.content}</div>}

      {msg.tool_calls?.map((tc, i) => (
        <div
          key={i}
          className="mt-2 flex items-center gap-1.5 rounded border bg-muted/50 px-2 py-1.5 text-[11px]"
        >
          <Wrench className="h-3 w-3 text-muted-foreground" />
          <Badge variant="outline" className="text-[10px]">
            {tc.name}
          </Badge>
          <span className="truncate text-muted-foreground">
            {JSON.stringify(tc.arguments).slice(0, 80)}
            {JSON.stringify(tc.arguments).length > 80 ? "..." : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

function ToolMessage({ msg, idx }: { msg: TraceMessage; idx: number }) {
  const isError = msg.tool_is_error;
  return (
    <div
      className={cn(
        "rounded-md border p-2.5",
        isError ? "border-destructive/30 bg-destructive/5" : "border-muted bg-muted/30",
      )}
    >
      <div className="mb-1 flex items-center gap-1.5">
        <Wrench className="h-3 w-3 text-muted-foreground" />
        {msg.tool_name && (
          <Badge variant="outline" className="text-[10px]">
            {msg.tool_name}
          </Badge>
        )}
        <Badge variant={isError ? "destructive" : "success"} className="text-[10px]">
          {isError ? "ERROR" : "VALID"}
        </Badge>
      </div>
      <Accordion>
        <AccordionItem value={`tool-${idx}`} className="!border-b-0">
          <AccordionTrigger className="py-1 text-[11px] text-muted-foreground">
            Result
          </AccordionTrigger>
          <AccordionContent>
            <pre className="max-h-32 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-[11px]">
              {msg.tool_result || msg.content}
            </pre>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}

export function LLMTracePanel({ trace }: { trace: LLMTrace }) {
  return (
    <div className="flex flex-col gap-2">
      <TraceSummary trace={trace} />
      {trace.messages.map((msg, i) => {
        switch (msg.role) {
          case "system":
            return <SystemMessage key={i} msg={msg} />;
          case "user":
            return <UserMessage key={i} msg={msg} />;
          case "assistant":
            return <AssistantMessage key={i} msg={msg} idx={i} />;
          case "tool":
            return <ToolMessage key={i} msg={msg} idx={i} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
