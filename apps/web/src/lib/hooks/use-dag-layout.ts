import {
  type DagGraphInput,
  type DagLayoutResult,
  EMPTY_LAYOUT,
  runDagLayout,
} from "@/lib/utils/dag-graph-layout";
import { useEffect, useMemo, useRef, useState } from "react";

interface UseDagLayoutResult extends DagLayoutResult {
  isLayouting: boolean;
}

/**
 * Async ELK layout for the bespoke DAG renderer. Mirrors the stale-guard
 * pattern of the legacy `useElkLayout`: re-layouts whenever the serialized
 * graph changes and ignores results that arrive after a newer input.
 */
export function useDagLayout(graph: DagGraphInput): UseDagLayoutResult {
  const [result, setResult] = useState<{ data: DagLayoutResult; key: string }>({
    data: EMPTY_LAYOUT,
    key: "",
  });

  const inputKey = useMemo(() => JSON.stringify(graph), [graph]);
  const latestKeyRef = useRef(inputKey);

  useEffect(() => {
    latestKeyRef.current = inputKey;
  });

  useEffect(() => {
    const currentKey = inputKey;
    runDagLayout(graph)
      .then((layout) => {
        if (latestKeyRef.current === currentKey) {
          setResult({ data: layout, key: currentKey });
        }
      })
      .catch((err: unknown) => {
        console.warn("DAG layout computation failed:", err);
      });
  }, [inputKey, graph]);

  const isLayouting = result.key !== inputKey;
  return { ...result.data, isLayouting };
}
