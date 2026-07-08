"use client";

import type { PipelineProgress } from "@/lib/hooks/use-run-events";
import { TRANSITION_META } from "@nof1-causal-lab/api-types";
import type { ArtifactViewId } from "@nof1-causal-lab/api-types";
import { ArrowDown } from "lucide-react";
import { motion } from "motion/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

/**
 * Fixed bottom notification for outputs that completed while the user
 * wasn't scrolled down to them. An output is "unseen" until the user
 * scrolls it into the viewport at least once. Scrolling back up does
 * NOT re-add already-seen outputs.
 */
export function CompletedOutputsNotification({ progress }: { progress: PipelineProgress }) {
  // Outputs the user has scrolled past (seen) at least once
  const seenRef = useRef<Set<ArtifactViewId>>(new Set());
  const [observerSeen, setObserverSeen] = useState<Set<ArtifactViewId>>(new Set());

  // Derive the list of completed/failed output ids from progress
  const completedArtifactIds = useMemo(() => {
    const ids: ArtifactViewId[] = [];
    for (const artifactId of progress.transitionOrder) {
      const status = progress.artifacts[artifactId];
      if (status === "completed" || status === "failed") {
        ids.push(artifactId);
      }
    }
    return ids;
  }, [progress]);

  // Unseen = completed outputs that haven't been scrolled into view
  const unseenIds = useMemo(
    () => completedArtifactIds.filter((id) => !observerSeen.has(id)),
    [completedArtifactIds, observerSeen],
  );

  useEffect(() => {
    const elementsToObserve: Element[] = [];

    for (const artifactId of completedArtifactIds) {
      if (seenRef.current.has(artifactId)) continue;
      const el = document.getElementById(artifactId);
      if (el) elementsToObserve.push(el);
    }

    if (elementsToObserve.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
      const newlySeen: ArtifactViewId[] = [];
      for (const entry of entries) {
        if (entry.isIntersecting) {
          const artifactId = entry.target.id as ArtifactViewId;
          seenRef.current.add(artifactId);
          observer.unobserve(entry.target);
          newlySeen.push(artifactId);
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
  }, [completedArtifactIds]);

  const scrollToNext = useCallback(() => {
    if (unseenIds.length === 0) return;
    const el = document.getElementById(unseenIds[0]);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [unseenIds]);

  if (unseenIds.length === 0) return null;

  const next = TRANSITION_META[unseenIds[0]];
  const label =
    unseenIds.length === 1
      ? `${next.label} completed`
      : `${unseenIds.length} new artifacts completed`;

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
