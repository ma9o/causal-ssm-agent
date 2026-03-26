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
import { Bot, User, Wrench } from "lucide-react";
import { memo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const remarkPlugins = [remarkGfm];

const TextPart = memo(function TextPart({ text }: { text: string }) {
  return (
    <div className="prose prose-sm max-w-none text-foreground prose-headings:text-foreground prose-strong:text-foreground prose-th:text-foreground prose-code:text-foreground prose-pre:bg-muted/50 prose-pre:text-foreground [&_pre]:text-xs [&_code]:text-xs [&_table]:text-xs [&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0 [&_h1]:text-base [&_h2]:text-sm [&_h3]:text-sm [&_h4]:text-sm [&_pre]:my-1 [&_pre]:p-2 [&_table]:block [&_table]:overflow-x-auto">
      <Markdown remarkPlugins={remarkPlugins}>{text}</Markdown>
    </div>
  );
});

function ReasoningPart({ text, idx }: { text: string; idx: number }) {
  return (
    <Accordion>
      <AccordionItem
        value={`reasoning-${idx}`}
        className="border-l-2 border-amber-400/50 pl-2.5 !border-b-0"
      >
        <AccordionTrigger className="py-1.5 text-xs text-amber-600">
          Thinking
        </AccordionTrigger>
        <AccordionContent>
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-xs">
            {text}
          </pre>
        </AccordionContent>
      </AccordionItem>
    </Accordion>
  );
}

function deepParseJson(value: unknown): unknown {
  if (typeof value === "string") {
    try {
      return deepParseJson(JSON.parse(value));
    } catch {
      return value;
    }
  }
  if (Array.isArray(value)) return value.map(deepParseJson);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([k, v]) => [k, deepParseJson(v)]),
    );
  }
  return value;
}

function formatToolData(data: unknown): string {
  if (typeof data === "string") return data;
  return JSON.stringify(deepParseJson(data), null, 2);
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
        <Wrench className="h-3.5 w-3.5 text-muted-foreground" />
        <Badge variant="outline" className="text-[11px]">
          {part.toolName}
        </Badge>
        {isFinished && (
          <Badge variant={hasError ? "destructive" : "success"} className="text-[11px]">
            {hasError ? "ERROR" : "OK"}
          </Badge>
        )}
        {!isFinished && (
          <span className="text-[11px] text-muted-foreground italic">pending</span>
        )}
      </div>

      {/* Input arguments */}
      {part.input != null && (
        <Accordion>
          <AccordionItem value={`tool-input-${idx}`} className="!border-b-0">
            <AccordionTrigger className="py-1 text-xs text-muted-foreground">
              Input
            </AccordionTrigger>
            <AccordionContent>
              <pre className="max-h-32 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-xs">
                {formatToolData(part.input)}
              </pre>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}

      {/* Output / Error */}
      {hasOutput && (
        <Accordion>
          <AccordionItem value={`tool-output-${idx}`} className="!border-b-0">
            <AccordionTrigger className="py-1 text-xs text-muted-foreground">
              Result
            </AccordionTrigger>
            <AccordionContent>
              <pre className="max-h-32 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-xs">
                {formatToolData(part.output)}
              </pre>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      )}
      {hasError && "errorText" in part && (
        <pre className="mt-1 max-h-32 overflow-auto whitespace-pre-wrap rounded bg-destructive/10 p-2 text-xs text-destructive">
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
        className="border-l-2 border-muted-foreground/30 pl-2.5 !border-b-0"
      >
        <AccordionTrigger className="py-2 text-xs text-muted-foreground">
          System prompt
        </AccordionTrigger>
        <AccordionContent>
          <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-xs">
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
      <div className="mb-1 flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-foreground">
        <User className="h-3.5 w-3.5" />
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
        <Bot className="h-3.5 w-3.5 text-primary" />
        <span className="text-[11px] font-medium uppercase tracking-wide text-primary">
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

export const ChatMessages = memo(function ChatMessages({ messages }: { messages: UIMessage[] }) {
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
});
