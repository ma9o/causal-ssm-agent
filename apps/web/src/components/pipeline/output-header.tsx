import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { TransitionRunStatus } from "@/lib/hooks/use-run-events";
import { cn } from "@/lib/utils";
import { linkifyDocRefs } from "@/lib/utils/linkify-docs";
import { Check, Circle, Loader2, X } from "lucide-react";

export function OutputHeader({
  title,
  status,
  context,
  staleArtifactIds,
}: {
  title: string;
  status: TransitionRunStatus;
  context?: string;
  /** Stale artifacts this transition produced — renders the amber "Stale" badge. */
  staleArtifactIds?: string[];
}) {
  const stale = (staleArtifactIds?.length ?? 0) > 0 && status !== "running";
  const staleDetail = `Inputs changed since this ran (stale: ${(staleArtifactIds ?? []).join(", ")}). Recompute to refresh.`;
  const StatusIcon =
    status === "completed"
      ? Check
      : status === "failed"
        ? X
        : status === "running"
          ? Loader2
          : Circle;

  return (
    <div className="flex items-center gap-3">
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full transition-colors",
          status === "completed"
            ? "bg-success text-success-foreground"
            : status === "failed"
              ? "bg-destructive text-white"
              : status === "running"
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground",
        )}
      >
        <StatusIcon className={cn("h-4 w-4", status === "running" && "animate-spin")} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-3">
          <h2 className="text-base font-semibold sm:text-lg">{title}</h2>
          {stale && (
            <Tooltip>
              <TooltipTrigger aria-label={staleDetail} className="cursor-default">
                <Badge variant="warning">Stale</Badge>
              </TooltipTrigger>
              <TooltipContent>{staleDetail}</TooltipContent>
            </Tooltip>
          )}
        </div>
        {context && (
          <p className="mt-0.5 text-sm text-muted-foreground">{linkifyDocRefs(context)}</p>
        )}
      </div>
    </div>
  );
}
