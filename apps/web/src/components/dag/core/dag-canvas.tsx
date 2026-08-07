import type { ComponentPropsWithoutRef, CSSProperties, ReactNode } from "react";
import { DAG_COLORS } from "./palette";

interface DagSvgProps
  extends Omit<ComponentPropsWithoutRef<"svg">, "width" | "height" | "viewBox"> {
  contentWidth: number;
  contentHeight: number;
  zoom: number;
}

export function DagCanvasFrame({ children }: { children: ReactNode }) {
  return <div style={CANVAS_FRAME}>{children}</div>;
}

export function DagSvg({
  contentWidth,
  contentHeight,
  zoom,
  children,
  style,
  ...svgProps
}: DagSvgProps) {
  return (
    <svg
      width={Math.ceil(contentWidth * zoom)}
      height={Math.ceil(contentHeight * zoom)}
      viewBox={`0 0 ${Math.ceil(contentWidth)} ${Math.ceil(contentHeight)}`}
      style={{ display: "block", ...style }}
      {...svgProps}
    >
      {children}
    </svg>
  );
}

const CANVAS_FRAME: CSSProperties = {
  background: "#fff",
  border: `1px solid ${DAG_COLORS.line}`,
  borderRadius: 14,
  padding: 6,
  minHeight: 560,
  maxHeight: "74vh",
  overflow: "auto",
  backgroundImage: `radial-gradient(${DAG_COLORS.line} .8px, transparent .8px)`,
  backgroundSize: "18px 18px",
};
