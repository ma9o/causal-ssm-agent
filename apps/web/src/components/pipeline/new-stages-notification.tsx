"use client";

import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { STAGES } from "@causal-ssm/api-types";
import type { StageId } from "@causal-ssm/api-types";
import { ArrowDown } from "lucide-react";
import { motion } from "motion/react";
import { useCallback, useEffect, useRef, useState } from "react";

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
  const [unseenIds, setUnseenIds] = useState<StageId[]>([]);

  useEffect(() => {
    const pendingUnseen: StageId[] = [];
    const elementsToObserve: { id: StageId; el: Element }[] = [];

    for (const s of STAGES) {
      const status = progress.stages[s.id];
      if (status !== "completed" && status !== "failed") continue;
      if (seenRef.current.has(s.id)) continue;

      const el = document.getElementById(s.id);
      if (!el) {
        pendingUnseen.push(s.id);
        continue;
      }
      elementsToObserve.push({ id: s.id, el });
    }

    if (elementsToObserve.length === 0) {
      setUnseenIds(pendingUnseen);
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      let changed = false;
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const stageId = entry.target.id as StageId;
          seenRef.current.add(stageId);
          observer.unobserve(entry.target);
          changed = true;
        }
      }
      if (changed) {
        setUnseenIds((prev) => prev.filter((id) => !seenRef.current.has(id)));
      }
    });

    // Check which elements are already visible vs need observation
    for (const { id, el } of elementsToObserve) {
      observer.observe(el);
    }

    // Elements not yet in DOM are unseen; observed elements start as unseen
    // until the observer fires (which is immediate if already in viewport)
    setUnseenIds([
      ...pendingUnseen,
      ...elementsToObserve
        .filter(({ id }) => !seenRef.current.has(id))
        .map(({ id }) => id),
    ]);

    return () => observer.disconnect();
  }, [progress]);

  const scrollToNext = useCallback(() => {
    if (unseenIds.length === 0) return;
    const el = document.getElementById(unseenIds[0]);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [unseenIds]);

  if (unseenIds.length === 0) return null;

  const next = STAGES.find((s) => s.id === unseenIds[0])!;
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
