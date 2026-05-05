"use client";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Panel, useReactFlow, useStore } from "@xyflow/react";
import { Minus, Plus } from "lucide-react";

const ZOOM_DURATION_MS = 120;

export const DAG_ZOOM_CONTROLS_INSET = {
  top: 48,
  left: 80,
} as const;

export function DagZoomControls() {
  const { zoomIn, zoomOut } = useReactFlow();
  const zoom = useStore((state) => state.transform[2]);
  const minZoom = useStore((state) => state.minZoom);
  const maxZoom = useStore((state) => state.maxZoom);

  const canZoomIn = zoom < maxZoom - 0.001;
  const canZoomOut = zoom > minZoom + 0.001;

  return (
    <Panel
      position="top-left"
      className="nodrag nopan flex gap-1 rounded-md border bg-card/90 p-1 shadow-sm backdrop-blur-sm"
    >
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label="Zoom in"
              disabled={!canZoomIn}
              onClick={() => {
                void zoomIn({ duration: ZOOM_DURATION_MS });
              }}
            >
              <Plus className="h-4 w-4" />
            </Button>
          }
        />
        <TooltipContent side="right">Zoom in</TooltipContent>
      </Tooltip>
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              aria-label="Zoom out"
              disabled={!canZoomOut}
              onClick={() => {
                void zoomOut({ duration: ZOOM_DURATION_MS });
              }}
            >
              <Minus className="h-4 w-4" />
            </Button>
          }
        />
        <TooltipContent side="right">Zoom out</TooltipContent>
      </Tooltip>
    </Panel>
  );
}
