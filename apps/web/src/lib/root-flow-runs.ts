export function dedupeRootFlowRunIds(rootFlowRunIds: readonly string[]): string[] {
  const deduped: string[] = [];

  for (const rootFlowRunId of rootFlowRunIds) {
    if (!rootFlowRunId || deduped.includes(rootFlowRunId)) {
      continue;
    }
    deduped.push(rootFlowRunId);
  }

  return deduped;
}

export function getLatestRootFlowRunId(rootFlowRunIds: readonly string[]): string | null {
  return rootFlowRunIds[rootFlowRunIds.length - 1] ?? null;
}

export function getWorkspaceRunTag(workspaceId: string): string {
  return `workspace:${workspaceId}`;
}
