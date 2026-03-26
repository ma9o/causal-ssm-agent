export function normalizeFlowRunIds(value: readonly unknown[] | null | undefined): string[] {
  if (!value) {
    return [];
  }

  const ids: string[] = [];
  const seen = new Set<string>();
  for (const item of value) {
    if (typeof item !== "string") {
      continue;
    }
    const normalized = item.trim();
    if (!normalized || seen.has(normalized)) {
      continue;
    }
    seen.add(normalized);
    ids.push(normalized);
  }

  return ids;
}

export function buildFlowRunIdsSignature(flowRunIds: readonly string[]): string {
  return normalizeFlowRunIds(flowRunIds).join("|");
}
