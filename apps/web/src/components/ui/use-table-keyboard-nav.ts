"use client";

import { useCallback, useState, type KeyboardEvent } from "react";

export function useTableKeyboardNav(rowCount: number) {
  const [focusedRowIndex, setFocusedRowIndex] = useState<number | null>(null);

  const onKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (rowCount === 0) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setFocusedRowIndex((prev) =>
            prev === null ? 0 : Math.min(prev + 1, rowCount - 1),
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setFocusedRowIndex((prev) =>
            prev === null ? 0 : Math.max(prev - 1, 0),
          );
          break;
        case "Home":
          e.preventDefault();
          setFocusedRowIndex(0);
          break;
        case "End":
          e.preventDefault();
          setFocusedRowIndex(rowCount - 1);
          break;
      }
    },
    [rowCount],
  );

  return {
    focusedRowIndex,
    containerProps: {
      tabIndex: 0,
      onKeyDown,
      role: "grid" as const,
    },
  };
}
