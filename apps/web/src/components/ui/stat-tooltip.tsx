import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";

export function StatTooltip({ explanation }: { explanation: string }) {
  return (
    <Tooltip>
      <TooltipTrigger
        render={<span />}
        className="inline-flex cursor-help"
      >
        <HelpCircle
          aria-hidden
          className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground transition-colors"
        />
      </TooltipTrigger>
      <TooltipContent>
        <span className="max-w-xs text-xs leading-relaxed">{explanation}</span>
      </TooltipContent>
    </Tooltip>
  );
}
