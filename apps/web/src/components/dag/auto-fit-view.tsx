"use client";

import {
  getViewportForBounds,
  useNodesInitialized,
  useReactFlow,
  useStore,
} from "@xyflow/react";
import { useEffect } from "react";

export function AutoFitView({
  fitViewKey,
  insets = { top: 0, right: 0, bottom: 0, left: 0 },
  padding = 0.08,
}: {
  fitViewKey: string;
  insets?: { top: number; right: number; bottom: number; left: number };
  padding?: number;
}) {
  const nodesInitialized = useNodesInitialized();
  const { getNodes, getNodesBounds, setViewport } = useReactFlow();
  const width = useStore((state) => state.width);
  const height = useStore((state) => state.height);
  const minZoom = useStore((state) => state.minZoom);
  const maxZoom = useStore((state) => state.maxZoom);

  useEffect(() => {
    if (!nodesInitialized || width <= 0 || height <= 0) return;

    const nodes = getNodes();
    if (nodes.length === 0) return;

    const availableWidth = Math.max(width - insets.left - insets.right, 1);
    const availableHeight = Math.max(height - insets.top - insets.bottom, 1);
    const bounds = getNodesBounds(nodes);
    const viewport = getViewportForBounds(
      bounds,
      availableWidth,
      availableHeight,
      minZoom,
      maxZoom,
      padding,
    );

    let frame: number | null = null;
    const timeout = window.setTimeout(() => {
      frame = requestAnimationFrame(() => {
        void setViewport(
          {
            x: viewport.x + insets.left,
            y: viewport.y + insets.top,
            zoom: viewport.zoom,
          },
          { duration: 0 },
        );
      });
    }, 80);

    return () => {
      window.clearTimeout(timeout);
      if (frame != null) {
        cancelAnimationFrame(frame);
      }
    };
  }, [
    fitViewKey,
    getNodes,
    getNodesBounds,
    height,
    insets.bottom,
    insets.left,
    insets.right,
    insets.top,
    maxZoom,
    minZoom,
    nodesInitialized,
    padding,
    setViewport,
    width,
  ]);

  return null;
}
