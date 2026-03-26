"use client";

import {
  refinementNeedsActivation,
  useRefinement,
} from "@/lib/contexts/refinement-context";
import type { RefinementUIMessage } from "@/lib/utils/trace-to-core";
import { useChat } from "@ai-sdk/react";
import type { LLMTrace, StageId } from "@causal-ssm/api-types";
import { INTERACTIVE_STAGES } from "@causal-ssm/api-types";
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
    return new DefaultChatTransport<RefinementUIMessage>({
      api: "/api/refine",
      body: { workspaceId, stageId, pendingStagePatch },
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
    <LLMTracePanelView
      trace={trace}
      refinementMessages={refinementMessages}
      canRefine={canRefine}
      isLoading={isLoading}
      input={input}
      onInputChange={setInput}
      onSubmit={handleSubmit}
    />
  );
}
