"use client";

import type { RefineApplyResponse } from "@/lib/api/analysis";
import { useRefinement } from "@/lib/contexts/refinement-context";
import { STAGES, STAGE_IDS } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import { Loader2, Play } from "lucide-react";
import { motion } from "motion/react";
import { useCallback, useState } from "react";

/**
 * Bottom-of-feed CTA that materializes the current interactive result.
 *
 * Earlier stages trigger a replay from the next stage boundary.
 * Terminal Stage 6 persists the finalized interactive result in place.
 */
export function ResumeButton({
  workspaceId,
  stageId,
  rootFlowRunId,
}: {
  workspaceId: string;
  stageId: string;
  rootFlowRunId?: string | null;
}) {
  const [applying, setApplying] = useState(false);
  const {
    clearPendingMaterialization,
    pendingStagePatches,
    refinementMessages,
  } = useRefinement();
  const normalizedStageId = stageId as StageId;

  const nextStageIdx = STAGE_IDS.indexOf(stageId as (typeof STAGE_IDS)[number]) + 1;
  const nextStage = STAGES[nextStageIdx];
  const isTerminalStage = nextStage == null;
  const pendingStagePatch = pendingStagePatches[normalizedStageId] ?? {};
  const pendingMessages = refinementMessages[normalizedStageId] ?? [];
  const resumeLabel = isTerminalStage
    ? "Persist Final Results"
    : `Apply Changes and Re-run from Stage ${nextStage.number}`;

  const handleResume = useCallback(async () => {
    if (applying) return;
    setApplying(true);
    try {
      const res = await fetch("/api/refine/apply", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          workspaceId,
          stageId,
          stagePatch: pendingStagePatch,
          messages: pendingMessages,
          ...(rootFlowRunId ? { rootFlowRunId } : {}),
        }),
      });

      if (!res.ok) {
        console.error("Resume failed:", await res.text());
        return;
      }

      const result = (await res.json()) as RefineApplyResponse;
      if (result.ok) {
        clearPendingMaterialization(normalizedStageId);
        if (isTerminalStage) {
          window.location.reload();
          return;
        }

        if (!result.rootFlowRunId) {
          console.error("Resume failed: missing rootFlowRunId in replay response");
          return;
        }

        window.location.href = `/analysis/${workspaceId}?${new URLSearchParams({
          rootFlowRunId: result.rootFlowRunId,
        }).toString()}`;
      }
    } finally {
      setApplying(false);
    }
  }, [
    workspaceId,
    stageId,
    rootFlowRunId,
    pendingStagePatch,
    pendingMessages,
    clearPendingMaterialization,
    normalizedStageId,
    applying,
    isTerminalStage,
  ]);

  return (
    <motion.div
      className="sticky bottom-6 mx-auto max-w-2xl px-4"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
    >
      <button
        type="button"
        onClick={handleResume}
        disabled={applying}
        className="flex w-full items-center justify-center gap-2 rounded-lg bg-green-600 px-6 py-3 text-sm font-semibold text-white shadow-lg transition-colors hover:bg-green-700 disabled:opacity-60"
      >
        {applying ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            {isTerminalStage ? "Persisting final results..." : "Applying changes and re-running..."}
          </>
        ) : (
          <>
            <Play className="h-4 w-4" />
            {resumeLabel}
          </>
        )}
      </button>
    </motion.div>
  );
}
