"use client";

import { useRefinement } from "@/lib/contexts/refinement-context";
import { STAGES, STAGE_IDS } from "@causal-ssm/api-types";
import { AlertTriangle } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";

export function InvalidationWarningModal() {
  const { modalOpen, pendingStageId, confirmRefinement, cancelRefinement } = useRefinement();

  if (!pendingStageId) return null;

  const stageIdx = STAGE_IDS.indexOf(pendingStageId);
  const pendingStage = STAGES.find((s) => s.id === pendingStageId);
  const invalidatedStages = STAGES.filter((_, i) => i > stageIdx);

  return (
    <AnimatePresence>
      {modalOpen && (
        <motion.div
          className="fixed inset-0 z-50 flex items-center justify-center"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          {/* Backdrop */}
          <div
            className="absolute inset-0 bg-black/50"
            onClick={cancelRefinement}
            onKeyDown={(e) => {
              if (e.key === "Escape") cancelRefinement();
            }}
          />

          {/* Dialog */}
          <motion.div
            className="relative z-10 mx-4 w-full max-w-md rounded-lg border bg-card p-6 shadow-lg"
            initial={{ scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
          >
            <div className="flex items-start gap-3">
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-amber-500/10">
                <AlertTriangle className="h-5 w-5 text-amber-500" />
              </div>
              <div className="min-w-0">
                <h3 className="text-base font-semibold">Invalidate downstream stages?</h3>
                <p className="mt-1.5 text-sm text-muted-foreground">
                  Editing{" "}
                  <span className="font-medium text-foreground">
                    Stage {pendingStage?.number} — {pendingStage?.label}
                  </span>{" "}
                  will invalidate all downstream results. They will need to be re-computed after
                  your changes.
                </p>
                {invalidatedStages.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {invalidatedStages.map((s) => (
                      <span
                        key={s.id}
                        className="inline-flex items-center rounded-md bg-muted px-2 py-0.5 text-xs text-muted-foreground"
                      >
                        {s.number}. {s.label}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="mt-5 flex justify-end gap-2">
              <button
                type="button"
                onClick={cancelRefinement}
                className="rounded-md border px-4 py-2 text-sm font-medium transition-colors hover:bg-muted"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={confirmRefinement}
                className="rounded-md bg-amber-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-amber-700"
              >
                Continue & Invalidate
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
