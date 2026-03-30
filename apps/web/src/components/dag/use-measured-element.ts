"use client";

import { useCallback, useEffect, useState } from "react";

type ElementSize = {
  width: number;
  height: number;
};

const EMPTY_SIZE: ElementSize = { width: 0, height: 0 };

export function useMeasuredElement<T extends HTMLElement>() {
  const [element, setElement] = useState<T | null>(null);
  const [size, setSize] = useState<ElementSize>(EMPTY_SIZE);

  const setRef = useCallback((node: T | null) => {
    setElement(node);
    setSize(
      node
        ? {
            width: node.offsetWidth,
            height: node.offsetHeight,
          }
        : EMPTY_SIZE,
    );
  }, []);

  useEffect(() => {
    if (!element) return;

    const measure = () => {
      setSize({
        width: element.offsetWidth,
        height: element.offsetHeight,
      });
    };

    measure();

    const observer = new ResizeObserver(measure);
    observer.observe(element);

    return () => observer.disconnect();
  }, [element]);

  return [setRef, size] as const;
}
