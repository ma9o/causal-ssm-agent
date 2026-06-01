"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import {
  isSuggestionsDataPart,
  type SuggestionAction,
  type SuggestionChip,
} from "@/lib/utils/trace-to-core";
import type {
  SimulateCounterfactualResult,
  SimulateInterventionResult,
} from "@nof1-causal-lab/api-types";
import type { UIMessage } from "ai";
import { Bot, Check, Eye, User, Wrench } from "lucide-react";
import { memo } from "react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

const remarkPlugins = [remarkGfm];
type DynamicToolMessagePart = Extract<UIMessage["parts"][number], { type: "dynamic-tool" }>;
type StaticToolMessagePart = Extract<UIMessage["parts"][number], { type: `tool-${string}` }>;
type ToolMessagePart = DynamicToolMessagePart | StaticToolMessagePart;

export type SimulationResult = SimulateInterventionResult | SimulateCounterfactualResult;

const SIMULATION_TOOLS = new Set(["simulate_intervention", "simulate_counterfactual"]);

function asSimulationResult(output: unknown): SimulationResult | null {
  if (typeof output !== "object" || output === null) return null;
  const candidate = output as { rung?: unknown; outcome?: unknown };
  return (candidate.rung === 2 || candidate.rung === 3) && typeof candidate.outcome === "string"
    ? (output as SimulationResult)
    : null;
}

function simulationHeadline(result: SimulationResult): string {
  const { mean } = result.summary;
  return `${mean >= 0 ? "+" : ""}${mean.toFixed(2)} SD on ${result.outcome}`;
}

const TextPart = memo(function TextPart({
  text,
  streaming = false,
}: {
  text: string;
  streaming?: boolean;
}) {
  if (streaming) {
    return <div className="whitespace-pre-wrap text-sm leading-6 text-foreground">{text}</div>;
  }

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

function isToolMessagePart(part: UIMessage["parts"][number]): part is ToolMessagePart {
  return part.type === "dynamic-tool" || part.type.startsWith("tool-");
}

function getToolName(part: ToolMessagePart): string {
  return part.type === "dynamic-tool" ? part.toolName : part.type.slice(5);
}

function ToolPart({
  part,
  idx,
  className,
  selected,
  onSelect,
  headline,
}: {
  part: ToolMessagePart;
  idx: number;
  className?: string;
  selected?: boolean;
  onSelect?: () => void;
  headline?: string;
}) {
  const hasOutput = part.state === "output-available";
  const hasError = part.state === "output-error";
  const isFinished = hasOutput || hasError;
  const toolName = getToolName(part);

  return (
    <div
      className={cn(
        "rounded-md border p-2.5",
        className,
        hasError ? "border-destructive/30 bg-destructive/5" : "border-muted bg-muted/30",
      )}
    >
      <div className="flex items-center gap-1.5">
        <Wrench className="h-3.5 w-3.5 text-muted-foreground" />
        <Badge variant="outline" className="text-[11px]">
          {toolName}
        </Badge>
        {isFinished && (
          <Badge variant={hasError ? "destructive" : "success"} className="text-[11px]">
            {hasError ? "ERROR" : "OK"}
          </Badge>
        )}
        {!isFinished && (
          <span className="text-[11px] text-muted-foreground italic">pending</span>
        )}
        {onSelect ? (
          <button
            type="button"
            onClick={onSelect}
            className={cn(
              "ml-auto inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium transition-colors",
              selected
                ? "bg-primary text-primary-foreground"
                : "border text-muted-foreground hover:text-foreground",
            )}
          >
            {selected ? (
              <>
                <Check className="h-3 w-3" /> Viewing
              </>
            ) : (
              <>
                <Eye className="h-3 w-3" /> View
              </>
            )}
          </button>
        ) : null}
      </div>
      {headline ? (
        <div className="mt-1 font-mono text-[11px] text-muted-foreground">{headline}</div>
      ) : null}

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

function SuggestionChips({
  suggestions,
  onSuggestionClick,
}: {
  suggestions: SuggestionChip[];
  onSuggestionClick?: (action: SuggestionAction, chip: SuggestionChip) => void;
}) {
  if (suggestions.length === 0) return null;
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {suggestions.map((chip, i) => (
        <Button
          key={`${chip.action.tool}-${i}`}
          type="button"
          variant="outline"
          size="xs"
          disabled={!onSuggestionClick}
          onClick={() => onSuggestionClick?.(chip.action, chip)}
        >
          {chip.label}
        </Button>
      ))}
    </div>
  );
}

function AssistantMessage({
  msg,
  streaming = false,
  onSuggestionClick,
  selectedSimulationKey,
  onSelectSimulation,
}: {
  msg: UIMessage;
  streaming?: boolean;
  onSuggestionClick?: (action: SuggestionAction, chip: SuggestionChip) => void;
  selectedSimulationKey?: string;
  onSelectSimulation?: (key: string, result: SimulationResult) => void;
}) {
  const suggestions = msg.parts
    .filter(isSuggestionsDataPart)
    .flatMap((part) => part.data.suggestions);

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
        if (isSuggestionsDataPart(part)) return null;
        switch (part.type) {
          case "text":
            return <TextPart key={key} text={part.text} streaming={streaming} />;
          case "reasoning":
            return <ReasoningPart key={key} text={part.text} idx={i} />;
          case "dynamic-tool": {
            const simulation =
              part.state === "output-available" && SIMULATION_TOOLS.has(part.toolName)
                ? asSimulationResult(part.output)
                : null;
            if (simulation && onSelectSimulation) {
              const callKey = part.toolCallId;
              return (
                <ToolPart
                  key={key}
                  part={part}
                  idx={i}
                  className="mt-2"
                  selected={callKey === selectedSimulationKey}
                  onSelect={() => onSelectSimulation(callKey, simulation)}
                  headline={simulationHeadline(simulation)}
                />
              );
            }
            return <ToolPart key={key} part={part} idx={i} className="mt-2" />;
          }
          case "tool-validate_measurement_model":
          case "tool-stage1b_grounding":
          case "tool-search_literature":
          case "tool-elicit_prior_gmm":
            return <ToolPart key={key} part={part} idx={i} className="mt-2" />;
          default:
            return isToolMessagePart(part) ? (
              <ToolPart key={key} part={part} idx={i} className="mt-2" />
            ) : null;
        }
      })}
      <SuggestionChips
        suggestions={suggestions}
        onSuggestionClick={onSuggestionClick}
      />
    </div>
  );
}

export const ChatMessages = memo(function ChatMessages({
  messages,
  streaming = false,
  onSuggestionClick,
  selectedSimulationKey,
  onSelectSimulation,
}: {
  messages: UIMessage[];
  streaming?: boolean;
  onSuggestionClick?: (action: SuggestionAction, chip: SuggestionChip) => void;
  selectedSimulationKey?: string;
  onSelectSimulation?: (key: string, result: SimulationResult) => void;
}) {
  return (
    <div className="flex flex-col gap-2">
      {messages.map((msg, idx) => {
        const isStreamingMessage =
          streaming && idx === messages.length - 1 && msg.role === "assistant";

        switch (msg.role) {
          case "system":
            return <SystemMessage key={msg.id} msg={msg} />;
          case "user":
            return <UserMessage key={msg.id} msg={msg} />;
          case "assistant":
            return (
              <AssistantMessage
                key={msg.id}
                msg={msg}
                streaming={isStreamingMessage}
                onSuggestionClick={onSuggestionClick}
                selectedSimulationKey={selectedSimulationKey}
                onSelectSimulation={onSelectSimulation}
              />
            );
          default:
            return null;
        }
      })}
    </div>
  );
});
