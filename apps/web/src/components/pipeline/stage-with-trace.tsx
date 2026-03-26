"use client";

import { useRefinement } from "@/lib/contexts/refinement-context";
import { cn } from "@/lib/utils";
import { Bot } from "lucide-react";
import { motion } from "motion/react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";

/**
 * Two-column layout: stage content on the left, LLM panel on the right.
 * Accepts `panelContent` as a ReactNode so the caller decides whether
 * to render the connected `LLMTracePanel` or the pure `LLMTracePanelView`.
 */
export function StageWithTrace({
  children,
  panelContent,
  stageId,
}: {
  children: ReactNode;
  panelContent: ReactNode;
  /** When provided, the panel auto-opens if a prefill targets this stage. */
  stageId?: string;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const leftRef = useRef<HTMLDivElement>(null);
  const [leftHeight, setLeftHeight] = useState<number | undefined>(undefined);
  const { prefill } = useRefinement();

  // Auto-open the LLM panel when a prefill targets this stage
  useEffect(() => {
    if (stageId && prefill?.stageId === stageId && !isOpen) {
      setIsOpen(true);
    }
  }, [prefill, stageId, isOpen]);

  const measureLeft = useCallback(() => {
    if (leftRef.current) setLeftHeight(leftRef.current.offsetHeight);
  }, []);

  useEffect(() => {
    if (!isOpen || !leftRef.current) return;
    measureLeft();
    const ro = new ResizeObserver(measureLeft);
    ro.observe(leftRef.current);
    return () => ro.disconnect();
  }, [isOpen, measureLeft]);

  const transition = { duration: 0.35, ease: [0.4, 0, 0.2, 1] as const };

  return (
    <div className={cn("flex", isOpen && "items-start gap-4")}>
      <motion.div
        ref={leftRef}
        className={cn("min-w-0", !isOpen && "max-w-6xl mx-auto w-full")}
        animate={{ flex: isOpen ? 2 : 1 }}
        transition={transition}
      >
        {!isOpen && (
          <div className="mb-2 flex justify-end">
            <button
              type="button"
              onClick={() => setIsOpen(true)}
              className="inline-flex items-center gap-1.5 rounded-md border border-muted bg-muted/50 px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted"
            >
              <Bot className="h-3.5 w-3.5" />
              Show Assistant Details
            </button>
          </div>
        )}
        {children}
      </motion.div>
      <motion.div
        className={cn("min-w-0", !isOpen && "h-0 overflow-hidden")}
        style={isOpen && leftHeight ? { height: leftHeight } : undefined}
        animate={{ flex: isOpen ? 1 : 0, opacity: isOpen ? 1 : 0 }}
        initial={false}
        transition={transition}
      >
        {isOpen && (
          <div className="flex h-full flex-col gap-3">
            <button
              type="button"
              onClick={() => setIsOpen(false)}
              className="inline-flex w-full shrink-0 items-center justify-center gap-1.5 rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-medium text-primary transition-colors"
            >
              <Bot className="h-3.5 w-3.5" />
              Hide Assistant Details
            </button>
            <div className="min-h-0 flex-1 flex flex-col rounded-lg border bg-muted/30 p-3">
              {panelContent}
            </div>
          </div>
        )}
      </motion.div>
    </div>
  );
}
