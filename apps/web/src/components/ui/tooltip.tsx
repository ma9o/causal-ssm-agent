"use client";

import { cn } from "@/lib/utils/cn";
import * as TooltipPrimitive from "@radix-ui/react-tooltip";
import { type ReactNode } from "react";

export function TooltipProvider({ children }: { children: ReactNode }) {
  return <TooltipPrimitive.Provider delayDuration={200}>{children}</TooltipPrimitive.Provider>;
}

export function Tooltip({
  children,
  content,
  className,
  triggerClassName,
}: {
  children: ReactNode;
  content: ReactNode;
  className?: string;
  triggerClassName?: string;
}) {
  return (
    <TooltipPrimitive.Root>
      <TooltipPrimitive.Trigger asChild>
        <div className={cn("inline-flex", triggerClassName)}>{children}</div>
      </TooltipPrimitive.Trigger>
      <TooltipPrimitive.Portal>
        <TooltipPrimitive.Content
          sideOffset={8}
          className={cn(
            "z-[100] rounded-md bg-popover px-3 py-1.5 text-sm text-popover-foreground shadow-md border animate-fade-in",
            "data-[state=closed]:animate-fade-out",
            className,
          )}
        >
          {content}
          <TooltipPrimitive.Arrow className="fill-popover" />
        </TooltipPrimitive.Content>
      </TooltipPrimitive.Portal>
    </TooltipPrimitive.Root>
  );
}
