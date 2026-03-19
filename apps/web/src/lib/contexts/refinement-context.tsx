"use client";

import { STAGE_IDS } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import { type ReactNode, createContext, useCallback, useContext, useMemo, useState } from "react";

interface RefinementState {
  /** Stage currently being refined (null if none). */
  refiningStageId: StageId | null;
  /** All stages after this index are visually invalidated. */
  invalidatedAfter: StageId | null;
  /** True when the refinement LLM is idle and has produced output. */
  settled: boolean;
  /** Whether the confirmation modal is open. */
  modalOpen: boolean;
  /** The stage that was requested for refinement (pending modal confirmation). */
  pendingStageId: StageId | null;

  /** Called by LLMTracePanel on first refinement message. Opens the modal. */
  requestRefinement: (stageId: StageId) => void;
  /** Called when user confirms the modal. */
  confirmRefinement: () => void;
  /** Called when user cancels the modal. */
  cancelRefinement: () => void;
  /** Called by LLMTracePanel to report whether the LLM has settled. */
  markSettled: (settled: boolean) => void;
  /** Check if a given stage is invalidated. */
  isInvalidated: (stageId: StageId) => boolean;
}

const RefinementContext = createContext<RefinementState | null>(null);

export function useRefinement() {
  const ctx = useContext(RefinementContext);
  if (!ctx) throw new Error("useRefinement must be used within RefinementProvider");
  return ctx;
}

export function RefinementProvider({ children }: { children: ReactNode }) {
  const [refiningStageId, setRefiningStageId] = useState<StageId | null>(null);
  const [invalidatedAfter, setInvalidatedAfter] = useState<StageId | null>(null);
  const [settled, setSettled] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [pendingStageId, setPendingStageId] = useState<StageId | null>(null);

  const requestRefinement = useCallback((stageId: StageId) => {
    setPendingStageId(stageId);
    setModalOpen(true);
  }, []);

  const confirmRefinement = useCallback(() => {
    if (pendingStageId) {
      setRefiningStageId(pendingStageId);
      setInvalidatedAfter(pendingStageId);
      setSettled(false);
    }
    setModalOpen(false);
    setPendingStageId(null);
  }, [pendingStageId]);

  const cancelRefinement = useCallback(() => {
    setModalOpen(false);
    setPendingStageId(null);
  }, []);

  const markSettled = useCallback((s: boolean) => setSettled(s), []);

  const isInvalidated = useCallback(
    (stageId: StageId) => {
      if (!invalidatedAfter) return false;
      const afterIdx = STAGE_IDS.indexOf(invalidatedAfter);
      const thisIdx = STAGE_IDS.indexOf(stageId);
      return thisIdx > afterIdx;
    },
    [invalidatedAfter],
  );

  const value = useMemo<RefinementState>(
    () => ({
      refiningStageId,
      invalidatedAfter,
      settled,
      modalOpen,
      pendingStageId,
      requestRefinement,
      confirmRefinement,
      cancelRefinement,
      markSettled,
      isInvalidated,
    }),
    [
      refiningStageId,
      invalidatedAfter,
      settled,
      modalOpen,
      pendingStageId,
      requestRefinement,
      confirmRefinement,
      cancelRefinement,
      markSettled,
      isInvalidated,
    ],
  );

  return <RefinementContext.Provider value={value}>{children}</RefinementContext.Provider>;
}
