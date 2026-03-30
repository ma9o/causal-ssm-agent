"use client";

import type { NodeProps } from "@xyflow/react";
import { memo } from "react";

interface NoiseNodeData {
  constructName: string;
  variance: number;
}

function NoiseNodeInner({ data }: NodeProps) {
  const d = data as unknown as NoiseNodeData;

  return (
    <div className="flex items-center justify-center rounded-md border border-dashed border-muted-foreground/40 bg-muted/50 px-2 py-1 text-[9px] text-muted-foreground font-mono backdrop-blur-sm">
      \u03C3\u00B2={d.variance.toFixed(2)}
    </div>
  );
}

export const NoiseNode = memo(NoiseNodeInner);
