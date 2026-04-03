import { type LayoutResult, layoutDag } from "@/lib/utils/dag-layout";
import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import { useEffect, useMemo, useRef, useState } from "react";

interface UseElkLayoutResult extends LayoutResult {
  isLayouting: boolean;
}

const EMPTY_RESULT: LayoutResult & { key: string } = { nodes: [], edges: [], key: "" };

export function useElkLayout(
  constructs: Construct[],
  causalEdges: CausalEdge[],
  indicators?: Indicator[],
): UseElkLayoutResult {
  const [result, setResult] = useState(EMPTY_RESULT);

  const inputKey = useMemo(
    () => JSON.stringify({ constructs, causalEdges, indicators }),
    [constructs, causalEdges, indicators],
  );

  const latestKeyRef = useRef(inputKey);

  useEffect(() => {
    latestKeyRef.current = inputKey;
  });

  useEffect(() => {
    const currentKey = inputKey;

    layoutDag(constructs, causalEdges, indicators)
      .then((layoutResult) => {
        if (latestKeyRef.current === currentKey) {
          setResult({ ...layoutResult, key: currentKey });
        }
      })
      .catch((err: unknown) => {
        console.warn("ELK layout computation failed:", err);
      });
  }, [inputKey, constructs, causalEdges, indicators]);

  const isLayouting = result.key !== inputKey;
  return { nodes: result.nodes, edges: result.edges, isLayouting };
}
