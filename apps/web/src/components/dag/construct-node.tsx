"use client";

import { Badge } from "@/components/ui/badge";
import { Tooltip } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils/cn";
import type { Construct, IdentifiedTreatmentStatus, Indicator } from "@causal-ssm/api-types";
import { Handle, type NodeProps, Position } from "@xyflow/react";
import { Star } from "lucide-react";
import { memo } from "react";

interface ConstructNodeData extends Construct {
  indicators?: Indicator[];
  identificationStatus?: "identified" | "non_identified";
  identificationDetails?: IdentifiedTreatmentStatus;
}

function IdentifiedTooltipContent({ details }: { details: IdentifiedTreatmentStatus }) {
  return (
    <div className="space-y-1.5 max-w-xs">
      <div className="flex items-center gap-2">
        <span className="text-muted-foreground text-xs">Method:</span>
        <Badge variant="success" className="text-[10px] px-1.5 py-0">
          {details.method}
        </Badge>
      </div>
      <div>
        <span className="text-muted-foreground text-xs">Estimand:</span>
        <code className="ml-1.5 rounded bg-muted px-1.5 py-0.5 text-[11px]">
          {details.estimand}
        </code>
      </div>
      {details.instruments.length > 0 && (
        <div className="flex flex-wrap items-center gap-1">
          <span className="text-muted-foreground text-xs">Instruments:</span>
          {details.instruments.map((inst) => (
            <Badge key={inst} variant="outline" className="text-[10px] px-1.5 py-0">
              {inst}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

function ConstructNodeInner({ data, selected }: NodeProps) {
  const construct = data as unknown as ConstructNodeData;
  const indicators = construct.indicators ?? [];

  const nodeContent = (
    <div
      className={cn(
        "rounded-lg border-2 shadow-sm transition-all duration-200 cursor-pointer",
        "hover:shadow-md hover:-translate-y-0.5",
        construct.identificationStatus === "identified"
          ? "bg-success/5"
          : construct.identificationStatus === "non_identified"
            ? "bg-destructive/5"
            : "bg-card",
        construct.role === "endogenous"
          ? "border-foreground/65"
          : "border-foreground/35",
        construct.is_outcome && "ring-2 ring-foreground/75 ring-offset-1",
        selected && "shadow-lg ring-2 ring-primary ring-offset-2",
      )}
    >
      <Handle type="target" position={Position.Top} className="!bg-muted-foreground !w-2 !h-2" />

      <div className="px-4 py-3">
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-semibold leading-tight">{construct.name}</span>
          {construct.is_outcome && (
            <Star className="h-3.5 w-3.5 shrink-0 fill-foreground/75 text-foreground/75" />
          )}
        </div>

        <div className="mt-1.5 flex flex-wrap gap-1">
          <Badge
            variant={construct.role === "endogenous" ? "default" : "secondary"}
            className="px-1.5 py-0 text-[10px]"
          >
            {construct.role === "endogenous" ? "endo" : "exo"}
          </Badge>
          <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
            {construct.temporal_status === "time_varying" ? "varying" : "invariant"}
          </Badge>
          {construct.temporal_scale && (
            <Badge variant="outline" className="px-1.5 py-0 text-[10px]">
              {construct.temporal_scale}
            </Badge>
          )}
        </div>
      </div>

      {indicators.length > 0 && (
        <div className="border-t border-dashed border-border px-3 py-1.5">
          {indicators.map((ind) => (
            <div key={ind.name} className="flex items-center justify-between gap-2 py-0.5">
              <span className="text-[11px] text-muted-foreground truncate">
                {ind.name}
              </span>
              <span className="text-[9px] text-muted-foreground shrink-0">
                {ind.measurement_dtype}
              </span>
            </div>
          ))}
        </div>
      )}

      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-muted-foreground !w-2 !h-2"
      />
    </div>
  );

  if (construct.identificationStatus === "identified" && construct.identificationDetails) {
    return (
      <Tooltip content={<IdentifiedTooltipContent details={construct.identificationDetails} />}>
        {nodeContent}
      </Tooltip>
    );
  }

  return nodeContent;
}

export const ConstructNode = memo(ConstructNodeInner);
