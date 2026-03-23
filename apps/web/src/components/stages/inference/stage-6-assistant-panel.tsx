"use client";

import { ChatMessages } from "@/components/ui/custom/chat-messages";
import type { UIMessage } from "ai";
import { Loader2, Send, Sparkles } from "lucide-react";
import { Suspense, lazy, type FormEvent, type ReactNode } from "react";

const Stage6AssistantLivePanel = lazy(() =>
  import("./stage-6-assistant-live-panel").then((module) => ({
    default: module.Stage6AssistantLivePanel,
  })),
);

export const EXAMPLE_PROMPTS = [
  {
    label: "Summarize Model",
    prompt:
      "Summarize the outcome, identifiable treatments, key diagnostics, and the strongest baseline stage-6 effects.",
  },
  {
    label: "Run Rung 2",
    prompt:
      "Pick the strongest identifiable treatment and run a rung 2 trajectory simulation over 30 days.",
  },
  {
    label: "Run Rung 3",
    prompt:
      "Using a recent observed window as evidence, run a rung 3 counterfactual for the strongest treatment and compare the final forecast.",
  },
];

export type Stage6AssistantStatus = "ready" | "submitted" | "streaming" | "error";

export interface Stage6AssistantDemoState {
  messages: UIMessage[];
  status?: "ready" | "submitted" | "streaming";
  showExamplePrompts?: boolean;
}

export function Stage6AssistantCard({
  messages,
  status,
  input,
  onInputChange,
  onSubmit,
  onExamplePrompt,
  interactionsDisabled = false,
  showExamplePrompts,
  messageFooter,
}: {
  messages: UIMessage[];
  status: Stage6AssistantStatus;
  input: string;
  onInputChange?: (value: string) => void;
  onSubmit?: (event: FormEvent) => void;
  onExamplePrompt?: (prompt: string) => void;
  interactionsDisabled?: boolean;
  showExamplePrompts: boolean;
  messageFooter?: ReactNode;
}) {
  const isLoading = status === "streaming" || status === "submitted";

  return (
    <div className="rounded-xl border bg-background/70 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Sparkles className="h-4 w-4 text-primary" />
            Stage 6 Assistant
          </div>
          <p className="max-w-2xl text-xs text-muted-foreground">
            Read-only chat over the fitted model. It can inspect the model, run Pearl rung 2
            intervention simulations, and run Pearl rung 3 counterfactual forecasts from observed
            history windows.
          </p>
        </div>
        <div className="rounded-full border px-2.5 py-1 text-[10px] font-medium uppercase tracking-wide text-muted-foreground">
          Read-only
        </div>
      </div>

      {showExamplePrompts && (
        <div className="mt-4 grid gap-2 sm:grid-cols-3">
          {EXAMPLE_PROMPTS.map((example) => (
            <button
              key={example.label}
              type="button"
              onClick={() => onExamplePrompt?.(example.prompt)}
              disabled={isLoading || interactionsDisabled || !onExamplePrompt}
              className="rounded-lg border bg-muted/30 p-3 text-left text-xs transition-colors hover:bg-muted/50 disabled:opacity-50"
            >
              <div className="mb-1 font-medium">{example.label}</div>
              <div className="text-muted-foreground">{example.prompt}</div>
            </button>
          ))}
        </div>
      )}

      <div className="mt-4 max-h-[28rem] overflow-y-auto">
        {messages.length > 0 ? (
          <ChatMessages messages={messages} />
        ) : (
          <div className="rounded-lg border border-dashed p-4 text-xs text-muted-foreground">
            Ask which interventions are identifiable, request a trajectory simulation, or condition
            on an observed window for a counterfactual forecast.
          </div>
        )}
        {messageFooter}
      </div>

      <form onSubmit={onSubmit} className="mt-4 flex gap-2">
        <input
          value={input}
          onChange={(event) => onInputChange?.(event.target.value)}
          placeholder="Ask about interventions or counterfactuals..."
          disabled={isLoading || interactionsDisabled || !onInputChange}
          className="flex-1 rounded-md border bg-background px-3 py-2 text-sm placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/30 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={isLoading || interactionsDisabled || !input.trim() || !onSubmit}
          className="inline-flex items-center gap-1.5 rounded-md bg-primary px-3 py-2 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary/90 disabled:opacity-50"
        >
          {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
        </button>
      </form>
    </div>
  );
}

export function Stage6AssistantPanel({
  userId,
  demoState,
}: {
  userId: string;
  demoState?: Stage6AssistantDemoState;
}) {
  if (demoState) {
    return (
      <Stage6AssistantCard
        messages={demoState.messages}
        status={demoState.status ?? "ready"}
        input=""
        interactionsDisabled
        showExamplePrompts={demoState.showExamplePrompts ?? demoState.messages.length === 0}
      />
    );
  }

  return (
    <Suspense
      fallback={
        <Stage6AssistantCard
          messages={[]}
          status="ready"
          input=""
          interactionsDisabled
          showExamplePrompts
        />
      }
    >
      <Stage6AssistantLivePanel userId={userId} />
    </Suspense>
  );
}
