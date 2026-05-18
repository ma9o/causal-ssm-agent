"use client";

import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@nof1-causal-lab/api-types";
import type { StageId } from "@nof1-causal-lab/api-types";
import { ArrowDown } from "lucide-react";
import { motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * Fixed bottom notification for stages that completed while the user
 * wasn't scrolled down to them. A stage is "unseen" until the user
 * scrolls it into the viewport at least once. Scrolling back up does
 * NOT re-add already-seen stages.
 */
export function NewStagesNotification({
  progress,
}: {
  progress: PipelineProgress;
}) {
  // Stages the user has scrolled past (seen) at least once
  const seenRef = useRef<Set<StageId>>(new Set());
  const [observerSeen, setObserverSeen] = useState<Set<StageId>>(new Set());

  // Derive the list of completed/failed stage IDs from progress
  const completedStageIds = useMemo(() => {
    const ids: StageId[] = [];
    for (const s of STAGES) {
      const status = progress.stages[s.id];
      if (status === "completed" || status === "failed") {
        ids.push(s.id);
      }
    }
    return ids;
  }, [progress]);

  // Unseen = completed stages that haven't been scrolled into view
  const unseenIds = useMemo(
    () => completedStageIds.filter((id) => !observerSeen.has(id)),
    [completedStageIds, observerSeen],
  );

  useEffect(() => {
    const elementsToObserve: Element[] = [];

    for (const stageId of completedStageIds) {
      if (seenRef.current.has(stageId)) continue;
      const el = document.getElementById(stageId);
      if (el) elementsToObserve.push(el);
    }

    if (elementsToObserve.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
      const newlySeen: StageId[] = [];
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const stageId = entry.target.id as StageId;
          seenRef.current.add(stageId);
          observer.unobserve(entry.target);
          newlySeen.push(stageId);
        }
      }
      if (newlySeen.length > 0) {
        setObserverSeen(new Set(seenRef.current));
      }
    });

    for (const el of elementsToObserve) {
      observer.observe(el);
    }

    return () => observer.disconnect();
  }, [completedStageIds]);

  const scrollToNext = useCallback(() => {
    if (unseenIds.length === 0) return;
    const el = document.getElementById(unseenIds[0]);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [unseenIds]);

  if (unseenIds.length === 0) return null;

  const next = STAGES.find((s) => s.id === unseenIds[0]);
  if (!next) return null;
  const label =
    unseenIds.length === 1
      ? `Stage ${next.number}: ${next.label} completed`
      : `${unseenIds.length} new stages completed`;

  return (
    <motion.button
      type="button"
      onClick={scrollToNext}
      className="fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-2 rounded-full border bg-background px-4 py-2.5 shadow-lg transition-colors hover:bg-secondary"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
    >
      <ArrowDown className="h-4 w-4" />
      <span className="text-sm font-medium">{label}</span>
    </motion.button>
  );
}
