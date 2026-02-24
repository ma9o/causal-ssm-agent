import { Tooltip } from "@/components/ui/tooltip";
import { HelpCircle } from "lucide-react";

export function StatTooltip({ explanation }: { explanation: string }) {
  return (
    <Tooltip
      content={<span className="max-w-xs text-xs leading-relaxed">{explanation}</span>}
    >
      <HelpCircle className="inline h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground transition-colors cursor-help" />
    </Tooltip>
  );
}
