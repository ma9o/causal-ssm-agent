import { Badge } from "@/components/ui/badge";
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils/cn";
import { linkifyDocRefs } from "@/lib/utils/linkify-docs";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { GateOverride } from "@causal-ssm/api-types";
import { ShieldAlert, ShieldCheck } from "lucide-react";

export function StageHeader({
  number,
  title,
  status,
  hasGate = false,
  context,
  gateOverridden,
  gateFailed = false,
}: {
  number: string;
  title: string;
  status: StageRunStatus;
  hasGate?: boolean;
  context?: string;
  gateOverridden?: GateOverride;
  gateFailed?: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-sm font-bold transition-colors",
          gateFailed || gateOverridden
            ? "bg-destructive text-white"
            : status === "completed"
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
          {hasGate && !gateOverridden && !gateFailed && (
            <Tooltip content="This stage can halt the pipeline if checks fail">
              <ShieldCheck className="h-4 w-4 text-foreground/75" />
            </Tooltip>
          )}
          {gateOverridden && (
            <Tooltip content={gateOverridden.reason}>
              <Badge variant="warning" className="gap-1">
                <ShieldAlert className="h-3 w-3" />
                Gate Overridden
              </Badge>
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
