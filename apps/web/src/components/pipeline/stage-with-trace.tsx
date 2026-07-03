"use client";

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
  open,
  defaultOpen = false,
  onOpenChange,
}: {
  children: ReactNode;
  panelContent: ReactNode;
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
    <div className={cn("mx-auto flex w-full max-w-[1600px]", isOpen && "items-start gap-4")}>
      <motion.div
        ref={leftRef}
        className="min-w-0"
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
              Show LLM Trace
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
              Hide LLM Trace
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

