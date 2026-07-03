import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { cn } from "@/lib/utils";
import { linkifyDocRefs } from "@/lib/utils/linkify-docs";

export function StageHeader({
  number,
  title,
  status,
  context,
  staleArtifactIds,
}: {
  number: string;
  title: string;
  status: StageRunStatus;
  context?: string;
  /** Stale artifacts this stage produced — renders the amber "Stale" badge. */
  staleArtifactIds?: string[];
}) {
  const stale = (staleArtifactIds?.length ?? 0) > 0 && status !== "running";
  const staleDetail = `Inputs changed since this ran (stale: ${(staleArtifactIds ?? []).join(", ")}). Recompute to refresh.`;

  return (
    <div className="flex items-center gap-3">
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold transition-colors",
          status === "completed"
            ? "bg-success text-success-foreground"
            : status === "failed"
              ? "bg-destructive text-white"
              : status === "running"
                ? "bg-primary text-primary-foreground"
                : "bg-secondary text-secondary-foreground",
        )}
      >
        {number}
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
