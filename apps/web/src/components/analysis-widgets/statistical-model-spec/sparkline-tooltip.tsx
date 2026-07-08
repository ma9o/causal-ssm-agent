const TOOLTIP_LINE_HEIGHT = 11;
const TOOLTIP_CHAR_WIDTH = 5.4;
const TOOLTIP_PAD_X = 5;
const TOOLTIP_PAD_Y = 4;
const TOOLTIP_OFFSET = 8;

/**
 * In-SVG hover tooltip for the inline sparklines. Rendered inside the chart's
 * own viewBox (not an HTML overlay) so it can never be clipped by table-cell
 * overflow, and it flips/clamps to stay within the chart bounds.
 */
export function SparklineTooltip({
  anchorX,
  anchorY,
  lines,
  width,
  height,
}: {
  anchorX: number;
  anchorY: number;
  lines: string[];
  width: number;
  height: number;
}) {
  const longest = Math.max(...lines.map((line) => line.length));
  const boxWidth = longest * TOOLTIP_CHAR_WIDTH + TOOLTIP_PAD_X * 2;
  const boxHeight = lines.length * TOOLTIP_LINE_HEIGHT + TOOLTIP_PAD_Y * 2;

  const preferRight = anchorX + TOOLTIP_OFFSET + boxWidth <= width;
  const rawX = preferRight ? anchorX + TOOLTIP_OFFSET : anchorX - TOOLTIP_OFFSET - boxWidth;
  const boxX = Math.max(0, Math.min(rawX, width - boxWidth));
  const boxY = Math.max(0, Math.min(anchorY - boxHeight / 2, height - boxHeight));

  return (
    <g pointerEvents="none">
      <rect
        x={boxX}
        y={boxY}
        width={boxWidth}
        height={boxHeight}
        rx={3}
        fill="var(--popover)"
        stroke="var(--border)"
        opacity={0.97}
      />
      {lines.map((line, index) => (
        <text
          key={line}
          x={boxX + TOOLTIP_PAD_X}
          y={boxY + TOOLTIP_PAD_Y + TOOLTIP_LINE_HEIGHT * index + 8}
          fontSize={9}
          fill="var(--popover-foreground)"
        >
          {line}
        </text>
      ))}
    </g>
  );
}
