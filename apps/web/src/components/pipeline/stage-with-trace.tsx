"use client";

import { useRefinement } from "@/lib/contexts/refinement-context";
import { cn } from "@/lib/utils";
import { Bot } from "lucide-react";
import { motion } from "motion/react";
import { type ReactNode, useCallback, useEffect, useRef, useState } from "react";

function useControllableOpen({
  open,
  defaultOpen,
  onOpenChange,
}: {
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (nextOpen: boolean) => void;
}) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(defaultOpen ?? false);

  const isOpen = open ?? uncontrolledOpen;
  const setIsOpen = useCallback(
    (nextOpen: boolean) => {
      if (open == null) {
        setUncontrolledOpen(nextOpen);
      }
      onOpenChange?.(nextOpen);
    },
    [onOpenChange, open],
  );

  return { isOpen, setIsOpen };
}

/**
 * Pure presentational two-column layout: stage content on the left,
 * LLM panel on the right. The button, side panel, and animation all live
 * inside this boundary so stories can render the full interaction shell.
 */
export function StageWithTraceView({
  children,
  panelContent,
  interactive,
  open,
  defaultOpen = false,
  onOpenChange,
}: {
  children: ReactNode;
  panelContent: ReactNode;
  interactive?: boolean;
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (nextOpen: boolean) => void;
}) {
  const { isOpen, setIsOpen } = useControllableOpen({
    open,
    defaultOpen,
    onOpenChange,
  });
  const leftRef = useRef<HTMLDivElement>(null);
  const [leftHeight, setLeftHeight] = useState<number | undefined>(undefined);

  const measureLeft = useCallback(() => {
    if (leftRef.current) {
      setLeftHeight(leftRef.current.offsetHeight);
    }
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
              className="inline-flex items-center gap-1.5 rounded-md border border-muted bg-muted/50 px-3 py-1.5 text-xs font-medium text-muted-foreground shadow-sm transition-colors hover:bg-muted hover:shadow-md"
            >
              <Bot className="h-3.5 w-3.5" />
              {interactive ? "Interact with LLM" : "Show LLM Trace"}
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
              {interactive ? "Hide LLM Chat" : "Hide LLM Trace"}
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

/**
 * Connected shell that reacts to refinement prefill state at runtime.
 * This wrapper owns app state; `StageWithTraceView` owns presentation.
 */
export function StageWithTrace({
  children,
  panelContent,
  stageId,
  interactive,
  defaultOpen = false,
}: {
  children: ReactNode;
  panelContent: ReactNode;
  stageId?: string;
  interactive?: boolean;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const { prefill } = useRefinement();

  useEffect(() => {
    if (stageId && prefill?.stageId === stageId) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- prefill comes from external refinement context and should open the shell immediately when targeted.
      setIsOpen(true);
    }
  }, [prefill, stageId]);

  return (
    <StageWithTraceView
      interactive={interactive}
      open={isOpen}
      onOpenChange={setIsOpen}
      panelContent={panelContent}
    >
      {children}
    </StageWithTraceView>
  );
}
