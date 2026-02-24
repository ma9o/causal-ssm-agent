"use client";

import type { StageId } from "@causal-ssm/api-types";
import { useCallback, useEffect, useRef } from "react";

export function useKeyboardNav(visibleStageIds: StageId[]) {
  const currentIndex = useRef(-1);

  const scrollToStage = useCallback(
    (index: number) => {
      if (index < 0 || index >= visibleStageIds.length) return;
      currentIndex.current = index;
      const el = document.getElementById(visibleStageIds[index]);
      el?.scrollIntoView({ behavior: "smooth", block: "start" });
    },
    [visibleStageIds],
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
        scrollToStage(Math.min(currentIndex.current + 1, visibleStageIds.length - 1));
      } else if (e.key === "k") {
        e.preventDefault();
        scrollToStage(Math.max(currentIndex.current - 1, 0));
      }
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [visibleStageIds, scrollToStage]);
}
