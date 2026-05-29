"use client";

import {
  refinementNeedsActivation,
  useRefinement,
} from "@/lib/contexts/refinement-context";
import type {
  RefinementUIMessage,
  SuggestionAction,
  SuggestionChip,
} from "@/lib/utils/trace-to-core";
import { useChat } from "@ai-sdk/react";
import type { LLMTrace, StageId } from "@nof1-causal-lab/api-types";
import { INTERACTIVE_STAGES } from "@nof1-causal-lab/api-types";
import { DefaultChatTransport } from "ai";
import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { LLMTracePanelView } from "./llm-trace-panel-view";

export { LLMTracePanelView } from "./llm-trace-panel-view";

const EMPTY_STAGE_PATCH: Record<string, unknown> = {};
const EMPTY_REFINEMENT_MESSAGES: RefinementUIMessage[] = [];

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
  const [input, setInput] = useState("");
  const queuedMessageRef = useRef<string | null>(null);

  const normalizedStageId = stageId as StageId | undefined;
  const {
    readOnly,
    refiningStageId,
    pendingStagePatches,
    refinementMessages: savedRefinementMessages,
    prefill,
    requestRefinement,
    markSettled,
    setPendingMaterialization,
    clearPrefill,
  } = useRefinement();

  const canRefine =
    !readOnly && interactive && !!workspaceId && !!stageId && INTERACTIVE_STAGES.includes(stageId);
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
    return new DefaultChatTransport<RefinementUIMessage>({
      api: "/api/refine",
      body: { workspaceId, stageId, pendingStagePatch },
    });
  }, [workspaceId, stageId, canRefine, pendingStagePatch]);

  const {
    messages: refinementMessages,
    sendMessage,
    setMessages,
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

  // Report settled state to the context so the ResumeButton can appear
  useEffect(() => {
    if (canRefine && refiningStageId === stageId) {
      markSettled(hasRefinement && !isLoading);
    }
  }, [canRefine, refiningStageId, stageId, hasRefinement, isLoading, markSettled]);

  // Consume prefill: inject prompt text into the input field
  useEffect(() => {
    if (prefill && stageId && prefill.stageId === stageId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- prefill is external refinement context state that must populate this controlled input when targeted.
      setInput(prefill.prompt);
      clearPrefill();
    }
  }, [prefill, stageId, clearPrefill]);

  // Send queued message after refinement is confirmed via the modal
  useEffect(() => {
    if (refiningStageId === stageId && queuedMessageRef.current) {
      const text = queuedMessageRef.current;
      queuedMessageRef.current = null;
      // eslint-disable-next-line react-hooks/set-state-in-effect -- the modal confirmed refinement, so the queued controlled input should clear before dispatch.
      setInput("");
      sendMessage({ text });
    }
  }, [refiningStageId, stageId, sendMessage]);

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || isLoading || !canRefine) return;

    // First message enters refinement mode. Non-terminal stages open the invalidation modal.
    // Don't clear input yet — the user may cancel the modal.
    if (refinementNeedsActivation(stageId as StageId, refiningStageId)) {
      queuedMessageRef.current = text;
      requestRefinement(stageId as StageId);
      return;
    }

    // Stage is already in refinement mode, or it's terminal and can refine in place.
    setInput("");
    sendMessage({ text });
  }

  async function dispatchAction(action: SuggestionAction, chip: SuggestionChip) {
    if (!workspaceId || !stageId) return;

    const toolCallId = crypto.randomUUID();
    const userMessage: RefinementUIMessage = {
      id: `chip-user-${toolCallId}`,
      role: "user",
      parts: [{ type: "text", text: `[Action] ${chip.label}` }],
    };
    const pendingMessage: RefinementUIMessage = {
      id: `chip-assistant-${toolCallId}`,
      role: "assistant",
      parts: [
        {
          type: "dynamic-tool",
          toolCallId,
          toolName: action.tool,
          state: "input-available",
          input: action.input,
        },
      ],
    };
    setMessages((prev) => [...prev, userMessage, pendingMessage]);

    try {
      const response = await fetch("/api/refine/dispatch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId,
          stageId,
          tool: action.tool,
          input: action.input,
        }),
      });
      const payload = (await response.json()) as
        | { output: unknown }
        | { error: string };

      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id !== pendingMessage.id) return msg;
          return {
            ...msg,
            parts: msg.parts.map((part) => {
              if (part.type !== "dynamic-tool" || part.toolCallId !== toolCallId) {
                return part;
              }
              if (!response.ok || "error" in payload) {
                return {
                  type: "dynamic-tool" as const,
                  toolCallId: part.toolCallId,
                  toolName: part.toolName,
                  state: "output-error" as const,
                  input: part.input,
                  errorText:
                    "error" in payload
                      ? payload.error
                      : `Tool dispatch failed (HTTP ${response.status})`,
                };
              }
              return {
                type: "dynamic-tool" as const,
                toolCallId: part.toolCallId,
                toolName: part.toolName,
                state: "output-available" as const,
                input: part.input,
                output: payload.output,
              };
            }),
          };
        }),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "Tool dispatch failed";
      setMessages((prev) =>
        prev.map((msg) => {
          if (msg.id !== pendingMessage.id) return msg;
          return {
            ...msg,
            parts: msg.parts.map((part) =>
              part.type === "dynamic-tool" && part.toolCallId === toolCallId
                ? {
                    type: "dynamic-tool" as const,
                    toolCallId: part.toolCallId,
                    toolName: part.toolName,
                    state: "output-error" as const,
                    input: part.input,
                    errorText: message,
                  }
                : part,
            ),
          };
        }),
      );
    }
  }

  function handleSuggestionClick(action: SuggestionAction, chip: SuggestionChip) {
    if (isLoading || !canRefine) return;
    void dispatchAction(action, chip);
  }

  return (
    <LLMTracePanelView
      trace={trace}
      refinementMessages={refinementMessages}
      canRefine={canRefine}
      isLoading={isLoading}
      input={input}
      onInputChange={setInput}
      onSubmit={handleSubmit}
      onSuggestionClick={canRefine ? handleSuggestionClick : undefined}
    />
  );
}
