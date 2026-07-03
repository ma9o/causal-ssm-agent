import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { cn } from "@/lib/utils";
import { linkifyDocRefs } from "@/lib/utils/linkify-docs";

export function StageHeader({
  number,
  title,
  status,
  context,
}: {
  number: string;
  title: string;
  status: StageRunStatus;
  context?: string;
}) {
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
        </div>
        {context && (
          <p className="mt-0.5 text-sm text-muted-foreground">{linkifyDocRefs(context)}</p>
        )}
      </div>
    </div>
  );
}
