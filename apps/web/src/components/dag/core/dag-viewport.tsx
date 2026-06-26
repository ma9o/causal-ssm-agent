"use client";

import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Maximize2, Minus, Plus } from "lucide-react";
import { type ReactNode, useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

interface Transform {
  x: number;
  y: number;
  k: number;
}

interface DagViewportProps {
  /** Laid-out content size in graph units (from the layout result). */
  contentWidth: number;
  contentHeight: number;
  /**
   * Changing this refits the view (e.g. when a new layout arrives). Container
   * resizes do NOT refit, so a user's pan/zoom is preserved while they work.
   */
  fitSignal?: string | number;
  minZoom?: number;
  maxZoom?: number;
  padding?: number;
  className?: string;
  ariaLabel?: string;
  /** Fired when the empty background is tapped (a click, not a pan/drag). */
  onBackgroundClick?: () => void;
  children: ReactNode;
}

const clamp = (v: number, lo: number, hi: number): number => Math.max(lo, Math.min(hi, v));

/** A pan gesture under this many pixels of travel reads as a click. */
const CLICK_SLOP = 4;

/**
 * Bespoke pan / zoom / fit container for the DAG renderer — replaces the React
 * Flow viewport (and `AutoFitView` + `DagZoomControls`). Content is rendered as
 * SVG children inside a transformed `<g>`. Pan: drag the background. Zoom:
 * ctrl/⌘ + wheel (so it never hijacks page scroll), or the on-canvas buttons.
 */
export function DagViewport({
  contentWidth,
  contentHeight,
  fitSignal,
  minZoom = 0.2,
  maxZoom = 2.5,
  padding = 24,
  className,
  ariaLabel = "Causal graph",
  onBackgroundClick,
  children,
}: DagViewportProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [transform, setTransform] = useState<Transform>({ x: 0, y: 0, k: 1 });
  const [panning, setPanning] = useState(false);
  const panOrigin = useRef<{ x: number; y: number } | null>(null);
  const gestureStart = useRef<{ x: number; y: number } | null>(null);
  const movedRef = useRef(false);
  const fittedRef = useRef<string | null>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const measure = () => setSize({ w: el.clientWidth, h: el.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const fit = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (w <= 0 || h <= 0 || contentWidth <= 0 || contentHeight <= 0) return;
    const k = clamp(
      Math.min((w - 2 * padding) / contentWidth, (h - 2 * padding) / contentHeight),
      minZoom,
      maxZoom,
    );
    setTransform({ k, x: (w - contentWidth * k) / 2, y: (h - contentHeight * k) / 2 });
  }, [contentWidth, contentHeight, minZoom, maxZoom, padding]);

  // Auto-fit on first valid layout and whenever fitSignal / content size changes.
  useLayoutEffect(() => {
    if (size.w <= 0 || contentWidth <= 0) return;
    const sig = `${fitSignal ?? ""}:${contentWidth}x${contentHeight}`;
    if (fittedRef.current === sig) return;
    fittedRef.current = sig;
    fit();
  }, [size.w, size.h, contentWidth, contentHeight, fitSignal, fit]);

  const zoomAround = useCallback(
    (px: number, py: number, factor: number) => {
      setTransform((prev) => {
        const k = clamp(prev.k * factor, minZoom, maxZoom);
        const wx = (px - prev.x) / prev.k;
        const wy = (py - prev.y) / prev.k;
        return { k, x: px - wx * k, y: py - wy * k };
      });
    },
    [minZoom, maxZoom],
  );

  const onWheel = useCallback(
    (e: React.WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const rect = containerRef.current?.getBoundingClientRect();
      if (!rect) return;
      zoomAround(e.clientX - rect.left, e.clientY - rect.top, e.deltaY < 0 ? 1.1 : 1 / 1.1);
    },
    [zoomAround],
  );

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (e.button !== 0) return;
    if ((e.target as Element).closest("[data-dag-interactive]")) return;
    panOrigin.current = { x: e.clientX, y: e.clientY };
    gestureStart.current = { x: e.clientX, y: e.clientY };
    movedRef.current = false;
    setPanning(true);
    (e.currentTarget as Element).setPointerCapture(e.pointerId);
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    const origin = panOrigin.current;
    if (!origin) return;
    const dx = e.clientX - origin.x;
    const dy = e.clientY - origin.y;
    panOrigin.current = { x: e.clientX, y: e.clientY };
    const start = gestureStart.current;
    if (start && Math.hypot(e.clientX - start.x, e.clientY - start.y) > CLICK_SLOP) {
      movedRef.current = true;
    }
    setTransform((prev) => ({ ...prev, x: prev.x + dx, y: prev.y + dy }));
  }, []);

  const endPan = useCallback(
    (e: React.PointerEvent, fireClick: boolean) => {
      if (!panOrigin.current) return;
      panOrigin.current = null;
      gestureStart.current = null;
      setPanning(false);
      try {
        (e.currentTarget as Element).releasePointerCapture(e.pointerId);
      } catch {
        // pointer already released
      }
      if (fireClick && !movedRef.current) onBackgroundClick?.();
    },
    [onBackgroundClick],
  );

  const zoomByButton = useCallback(
    (factor: number) => {
      const rect = containerRef.current?.getBoundingClientRect();
      zoomAround((rect?.width ?? 0) / 2, (rect?.height ?? 0) / 2, factor);
    },
    [zoomAround],
  );

  return (
    <div ref={containerRef} className={cn("relative h-full w-full overflow-hidden", className)}>
      <svg
        width="100%"
        height="100%"
        role="img"
        aria-label={ariaLabel}
        onWheel={onWheel}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(e) => endPan(e, true)}
        onPointerLeave={(e) => endPan(e, false)}
        style={{ touchAction: "none", cursor: panning ? "grabbing" : "grab" }}
      >
        <g transform={`translate(${transform.x} ${transform.y}) scale(${transform.k})`}>
          {children}
        </g>
      </svg>

      <div className="absolute left-2 top-2 flex flex-col gap-1 rounded-md border bg-card/90 p-1 shadow-sm backdrop-blur-sm">
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Zoom in"
          onClick={() => zoomByButton(1.2)}
        >
          <Plus className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label="Zoom out"
          onClick={() => zoomByButton(1 / 1.2)}
        >
          <Minus className="h-4 w-4" />
        </Button>
        <Button type="button" variant="ghost" size="icon-sm" aria-label="Fit to view" onClick={fit}>
          <Maximize2 className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
