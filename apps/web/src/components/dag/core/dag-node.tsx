"use client";

import type { ReactNode } from "react";

interface DagNodeShellProps {
  width: number;
  height: number;
  title?: string;
  subtitle?: string;
  /** Border accent (e.g. sign color, status). Defaults to the theme border. */
  accent?: string;
  highlighted?: boolean;
  /** Outcome nodes get a heavier border. */
  outcome?: boolean;
  children?: ReactNode;
}

/**
 * The shared SVG card shell both DAG kinds build on: a rounded rect with an
 * optional title/subtitle. Variants render their body (badges, trajectory
 * charts, glyphs, do() controls) as children.
 */
export function DagNodeShell({
  width,
  height,
  title,
  subtitle,
  accent,
  highlighted,
  outcome,
  children,
}: DagNodeShellProps) {
  const strokeWidth = highlighted ? 3 : outcome ? 2 : 1.4;
  return (
    <>
      <rect
        width={width}
        height={height}
        rx={11}
        fill="var(--card)"
        stroke={accent ?? "var(--border)"}
        strokeWidth={strokeWidth}
        style={{ transition: "stroke 200ms" }}
      />
      {title ? (
        <text x={14} y={24} fontSize={13} fontWeight={650} fill="var(--foreground)">
          {title}
        </text>
      ) : null}
      {subtitle ? (
        <text x={14} y={40} fontSize={9.5} fill="var(--muted-foreground)">
          {subtitle}
        </text>
      ) : null}
      {children}
    </>
  );
}
