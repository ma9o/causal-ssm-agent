"use client";

import type { ArtifactViewId } from "@nof1-causal-lab/api-types";
import { useCallback, useEffect, useRef } from "react";

export function useKeyboardNav(visibleArtifactIds: ArtifactViewId[]) {
  const currentIndex = useRef(-1);

  const scrollToArtifact = useCallback(
    (index: number) => {
      if (index < 0 || index >= visibleArtifactIds.length) return;
      currentIndex.current = index;
      const el = document.getElementById(visibleArtifactIds[index]);
      el?.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [visibleArtifactIds],
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't capture when typing in inputs
      if (
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLInputElement ||
        (e.target as HTMLElement)?.isContentEditable
      ) {
        return;
      }

      if (e.key === "j") {
        e.preventDefault();
        scrollToArtifact(Math.min(currentIndex.current + 1, visibleArtifactIds.length - 1));
      } else if (e.key === "k") {
        e.preventDefault();
        scrollToArtifact(Math.max(currentIndex.current - 1, 0));
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [visibleArtifactIds, scrollToArtifact]);
}
