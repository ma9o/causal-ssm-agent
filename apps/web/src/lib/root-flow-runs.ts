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

export function mergeRootFlowRunIds(
  ...inputs: Array<readonly string[] | string | null | undefined>
): string[] {
  const merged: string[] = [];

  for (const input of inputs) {
    if (!input) {
      continue;
    }

    if (typeof input === "string") {
      merged.push(input);
      continue;
    }

    merged.push(...input);
  }

  return dedupeRootFlowRunIds(merged);
}

export function getLatestRootFlowRunId(rootFlowRunIds: readonly string[]): string | null {
  return rootFlowRunIds[rootFlowRunIds.length - 1] ?? null;
}
