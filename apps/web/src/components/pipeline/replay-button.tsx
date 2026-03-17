"use client";

import type { ReplayResponse } from "@/lib/api/analysis";
import { INTERACTIVE_STAGES } from "@causal-ssm/api-types";
import { Loader2, RotateCcw } from "lucide-react";
import { useCallback, useState } from "react";

/**
 * Replay button shown below completed interactive stage outputs.
 *
 * Click → POST /api/replay with current stage data → pipeline re-runs
 * all downstream stages → frontend navigates to new run.
 */
export function ReplayButton({
  userId,
  stageId,
}: {
  userId: string;
  stageId: string;
}) {
  const [replaying, setReplaying] = useState(false);

  const canReplay = INTERACTIVE_STAGES.includes(stageId);

  const handleReplay = useCallback(async () => {
    if (replaying) return;
    setReplaying(true);
    try {
      // Load current stage data
      const dataRes = await fetch(`/api/results/${userId}/${stageId}`);
      if (!dataRes.ok) return;
      const stageData = await dataRes.json();

      // Remove internal fields
      const { llm_trace: _, outcome: __, ...domainData } = stageData;

      const res = await fetch("/api/replay", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ userId, stageId, stageData: domainData }),
      });

      if (!res.ok) {
        console.error("Replay failed:", await res.text());
        return;
      }

      const result = (await res.json()) as ReplayResponse;
      if (result.ok) {
        window.location.href = `/analysis/${userId}`;
      }
    } finally {
      setReplaying(false);
    }
  }, [userId, stageId, replaying]);

  if (!canReplay) return null;

  return (
    <div className="mt-3 rounded-lg border-t border-dashed border-muted-foreground/30 p-3">
      <button
        type="button"
        onClick={handleReplay}
        disabled={replaying}
        className="inline-flex w-full items-center justify-center gap-2 rounded-md border bg-background px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-muted disabled:opacity-50"
      >
        {replaying ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Re-running downstream stages...
          </>
        ) : (
          <>
            <RotateCcw className="h-4 w-4" />
            Replay from this stage
          </>
        )}
      </button>
    </div>
  );
}
