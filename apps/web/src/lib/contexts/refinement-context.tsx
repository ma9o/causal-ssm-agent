"use client";

import { STAGE_IDS } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import type { RefinementUIMessage } from "@/lib/utils/trace-to-core";
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
  /** Pending stage patches returned by the refinement server for each stage. */
  pendingStagePatches: Partial<Record<StageId, Record<string, unknown>>>;
  /** In-memory refinement conversations that have not been materialized yet. */
  refinementMessages: Partial<Record<StageId, RefinementUIMessage[]>>;

  /** Called by LLMTracePanel on first refinement message. Opens the modal. */
  requestRefinement: (stageId: StageId) => void;
  /** Called when user confirms the modal. */
  confirmRefinement: () => void;
  /** Called when user cancels the modal. */
  cancelRefinement: () => void;
  /** Called by LLMTracePanel to report whether the LLM has settled. */
  markSettled: (settled: boolean) => void;
  /** Replace the in-memory materialization payload for a stage. */
  setPendingMaterialization: (
    stageId: StageId,
    payload: {
      messages?: RefinementUIMessage[];
      stagePatch?: Record<string, unknown>;
    },
  ) => void;
  /** Clear the in-memory materialization payload for a stage. */
  clearPendingMaterialization: (stageId: StageId) => void;
  /** Check if a given stage is invalidated. */
  isInvalidated: (stageId: StageId) => boolean;
}

const RefinementContext = createContext<RefinementState | null>(null);

export function stageHasDownstreamStages(stageId: StageId): boolean {
  const stageIdx = STAGE_IDS.indexOf(stageId);
  return stageIdx !== -1 && stageIdx < STAGE_IDS.length - 1;
}

export function refinementNeedsActivation(
  stageId: StageId,
  refiningStageId: StageId | null,
): boolean {
  return refiningStageId !== stageId;
}

export function refinementRequiresConfirmation(
  stageId: StageId,
  refiningStageId: StageId | null,
): boolean {
  return refinementNeedsActivation(stageId, refiningStageId) && stageHasDownstreamStages(stageId);
}

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
  const [pendingStagePatches, setPendingStagePatches] = useState<
    Partial<Record<StageId, Record<string, unknown>>>
  >({});
  const [refinementMessages, setRefinementMessages] = useState<
    Partial<Record<StageId, RefinementUIMessage[]>>
  >({});

  const requestRefinement = useCallback((stageId: StageId) => {
    if (!stageHasDownstreamStages(stageId)) {
      setRefiningStageId(stageId);
      setSettled(false);
      setPendingStageId(null);
      setModalOpen(false);
      return;
    }

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

  const setPendingMaterialization = useCallback(
    (
      stageId: StageId,
      payload: {
        messages?: RefinementUIMessage[];
        stagePatch?: Record<string, unknown>;
      },
    ) => {
      if (payload.messages) {
        setRefinementMessages((current) => ({
          ...current,
          [stageId]: payload.messages,
        }));
      }

      if (payload.stagePatch) {
        setPendingStagePatches((current) => ({
          ...current,
          [stageId]: payload.stagePatch,
        }));
      }
    },
    [],
  );

  const clearPendingMaterialization = useCallback((stageId: StageId) => {
    setRefinementMessages((current) => {
      const next = { ...current };
      delete next[stageId];
      return next;
    });
    setPendingStagePatches((current) => {
      const next = { ...current };
      delete next[stageId];
      return next;
    });
  }, []);

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
      pendingStagePatches,
      refinementMessages,
      requestRefinement,
      confirmRefinement,
      cancelRefinement,
      markSettled,
      setPendingMaterialization,
      clearPendingMaterialization,
      isInvalidated,
    }),
    [
      refiningStageId,
      invalidatedAfter,
      settled,
      modalOpen,
      pendingStageId,
      pendingStagePatches,
      refinementMessages,
      requestRefinement,
      confirmRefinement,
      cancelRefinement,
      markSettled,
      setPendingMaterialization,
      clearPendingMaterialization,
      isInvalidated,
    ],
  );

  return <RefinementContext.Provider value={value}>{children}</RefinementContext.Provider>;
}
