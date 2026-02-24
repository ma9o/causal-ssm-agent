import { layoutDag, type LayoutResult } from "@/lib/utils/dag-layout";
import type { CausalEdge, Construct, Indicator } from "@causal-ssm/api-types";
import { useEffect, useMemo, useRef, useState } from "react";

interface UseElkLayoutResult extends LayoutResult {
  isLayouting: boolean;
}

const EMPTY_RESULT: LayoutResult = { nodes: [], edges: [] };

export function useElkLayout(
  constructs: Construct[],
  causalEdges: CausalEdge[],
  indicators?: Indicator[],
): UseElkLayoutResult {
  const [result, setResult] = useState<LayoutResult>(EMPTY_RESULT);
  const [isLayouting, setIsLayouting] = useState(true);

  const inputKey = useMemo(
    () => JSON.stringify({ constructs, causalEdges, indicators }),
    [constructs, causalEdges, indicators],
  );

  const latestKeyRef = useRef(inputKey);
  latestKeyRef.current = inputKey;

  useEffect(() => {
    const currentKey = inputKey;
    setIsLayouting(true);

    layoutDag(constructs, causalEdges, indicators).then((layoutResult) => {
      if (latestKeyRef.current === currentKey) {
        setResult(layoutResult);
        setIsLayouting(false);
      }
    });
  }, [inputKey, constructs, causalEdges, indicators]);

  return { ...result, isLayouting };
}
