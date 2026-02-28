"use client";

import { type AllStageData, generateMarkdown } from "@/lib/utils/generate-markdown";
import { STAGE_IDS, type StageId } from "@causal-ssm/api-types";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback } from "react";

export function useExportMarkdown(runId: string) {
  const queryClient = useQueryClient();

  const exportToMarkdown = useCallback(() => {
    // Read all cached stage data (zero network requests)
    const allData: AllStageData = {};
    for (const stageId of STAGE_IDS) {
      const cached = queryClient.getQueryData<unknown>(["pipeline", runId, "stage", stageId]);
      if (cached) {
        (allData as Record<StageId, unknown>)[stageId] = cached;
      }
    }

    const markdown = generateMarkdown(allData, runId);

    // Trigger download
    const blob = new Blob([markdown], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `pipeline-report-${runId.slice(0, 8)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [runId, queryClient]);

  return { exportToMarkdown };
}
