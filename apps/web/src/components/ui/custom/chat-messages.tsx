"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { UIMessage } from "ai";
import { Bot, ChevronRight, Wrench } from "lucide-react";

function TextPart({ text }: { text: string }) {
  return <div className="whitespace-pre-wrap text-xs">{text}</div>;
}

function ReasoningPart({ text, idx }: { text: string; idx: number }) {
  return (
    <Accordion>
      <AccordionItem
        value={`reasoning-${idx}`}
        className="border-l-2 border-amber-400/50 !border-b-0"
      >
        <AccordionTrigger className="py-1.5 text-[11px] text-amber-600">
          Thinking
        </AccordionTrigger>
        <AccordionContent>
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-amber-50/50 p-2 text-[11px]">
            {text}
          </pre>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function ToolPart({
  part,
  idx,
}: {
  part: Extract<UIMessage["parts"][number], { type: "dynamic-tool" }>;
  idx: number;
}) {
  const hasOutput = part.state === "output-available";
  const hasError = part.state === "output-error";
  const isFinished = hasOutput || hasError;

  return (
    <div
      className={cn(
        "mt-2 rounded-md border p-2.5",
        hasError ? "border-destructive/30 bg-destructive/5" : "border-muted bg-muted/30",
      )}
    >
      <div className="flex items-center gap-1.5">
        <Wrench className="h-3 w-3 text-muted-foreground" />
        <Badge variant="outline" className="text-[10px]">
          {part.toolName}
        </Badge>
        {isFinished && (
          <Badge variant={hasError ? "destructive" : "success"} className="text-[10px]">
            {hasError ? "ERROR" : "OK"}
          </Badge>
        )}
        {!isFinished && (
          <span className="text-[10px] text-muted-foreground italic">pending</span>
        )}
      </div>

      {/* Input arguments */}
      {part.input != null && (
        <Accordion>
          <AccordionItem value={`tool-input-${idx}`} className="!border-b-0">
            <AccordionTrigger className="py-1 text-[11px] text-muted-foreground">
              Input
            </AccordionTrigger>
            <AccordionContent>
              <pre className="max-h-32 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-[11px]">
                {typeof part.input === "string"
                  ? part.input
                  : JSON.stringify(part.input, null, 2)}
              </pre>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {/* Output / Error */}
      {hasOutput && (
        <Accordion>
          <AccordionItem value={`tool-output-${idx}`} className="!border-b-0">
            <AccordionTrigger className="py-1 text-[11px] text-muted-foreground">
              Result
            </AccordionTrigger>
            <AccordionContent>
              <pre className="max-h-32 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-[11px]">
                {typeof part.output === "string"
                  ? part.output
                  : JSON.stringify(part.output, null, 2)}
              </pre>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
      {hasError && "errorText" in part && (
        <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-destructive/10 p-2 text-[11px] text-destructive">
          {part.errorText}
        </pre>
      )}
    </div>
  );
}

function SystemMessage({ msg }: { msg: UIMessage }) {
  const text = msg.parts.find((p) => p.type === "text");
  if (!text || text.type !== "text") return null;

  return (
    <Accordion>
      <AccordionItem
        value="system"
        className="border-l-2 border-muted-foreground/30 !border-b-0"
      >
        <AccordionTrigger className="py-2 text-xs text-muted-foreground">
          <span className="flex items-center gap-1.5">
            <ChevronRight className="h-3 w-3" />
            System prompt
          </span>
        </AccordionTrigger>
        <AccordionContent>
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-[11px]">
            {text.text}
          </pre>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function UserMessage({ msg }: { msg: UIMessage }) {
  const text = msg.parts.find((p) => p.type === "text");
  return (
    <div className="rounded-md border bg-background p-2.5">
      <div className="mb-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
        User
      </div>
      {text?.type === "text" && <TextPart text={text.text} />}
    </div>
  );
}

function AssistantMessage({ msg }: { msg: UIMessage }) {
  return (
    <div className="rounded-md border border-primary/20 bg-primary/5 p-2.5">
      <div className="mb-1 flex items-center gap-1.5">
        <Bot className="h-3 w-3 text-primary" />
        <span className="text-[10px] font-medium uppercase tracking-wide text-primary">
          Assistant
        </span>
      </div>
      {msg.parts.map((part, i) => {
        const key = `${part.type}-${i}`;
        switch (part.type) {
          case "text":
            return <TextPart key={key} text={part.text} />;
          case "reasoning":
            return <ReasoningPart key={key} text={part.text} idx={i} />;
          case "dynamic-tool":
            return <ToolPart key={key} part={part} idx={i} />;
          default:
            return null;
        }
      })}
    </div>
  );
}

export function ChatMessages({ messages }: { messages: UIMessage[] }) {
  return (
    <div className="flex flex-col gap-2">
      {messages.map((msg) => {
        switch (msg.role) {
          case "system":
            return <SystemMessage key={msg.id} msg={msg} />;
          case "user":
            return <UserMessage key={msg.id} msg={msg} />;
          case "assistant":
            return <AssistantMessage key={msg.id} msg={msg} />;
          default:
            return null;
        }
      })}
    </div>
  );
}
