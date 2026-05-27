"use client";

import type { StageMeta } from "@nof1-causal-lab/api-types";
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

const useIsoLayoutEffect = typeof window === "undefined" ? useEffect : useLayoutEffect;

export function LazyStageMount({
  stage,
  rootMarginPx = 300,
  minHeight = 400,
  children,
}: {
  stage: StageMeta;
  rootMarginPx?: number;
  minHeight?: number;
  children: ReactNode;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [mounted, setMounted] = useState(false);

  useIsoLayoutEffect(() => {
    if (mounted) return;
    if (window.location.hash === `#${stage.id}`) {
      setMounted(true);
      return;
    }
    const node = ref.current;
    if (!node) return;
    const rect = node.getBoundingClientRect();
    if (rect.top < window.innerHeight + rootMarginPx && rect.bottom > -rootMarginPx) {
      setMounted(true);
    }
  }, [stage.id, mounted, rootMarginPx]);

  useEffect(() => {
    if (mounted) return;
    const node = ref.current;
    if (!node) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setMounted(true);
          observer.disconnect();
        }
      },
      { rootMargin: `${rootMarginPx}px 0px` },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [mounted, rootMarginPx]);

  return (
    <div
      ref={ref}
      id={stage.id}
      data-lazy-stage={stage.id}
      data-lazy-mounted={mounted ? "true" : "false"}
      style={{ minHeight }}
      className="scroll-mt-28"
    >
      {mounted ? (
        children
      ) : (
        <section
          aria-busy="true"
          className="rounded-lg border border-dashed bg-card/40 p-4 text-sm text-muted-foreground shadow-sm sm:p-6"
          style={{ minHeight }}
        >
          <div className="flex items-center gap-3">
            <span className="rounded-full border px-2 py-0.5 text-xs font-medium">
              {stage.number}
            </span>
            <span className="font-medium">{stage.label}</span>
          </div>
        </section>
      )}
    </div>
  );
}
