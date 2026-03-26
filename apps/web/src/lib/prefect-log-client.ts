export interface PrefectLogEntry {
  id: string;
  created: string;
  name: string;
  level: number;
  message: string;
  timestamp: string;
  flow_run_id: string;
  task_run_id: string | null;
}

const LOG_PAGE_SIZE = 200;
const LOG_STREAM_LOOKBACK_MS = 60_000;
const LOG_STREAM_LOOKAHEAD_MS = 365 * 24 * 60 * 60 * 1000;

const LOG_LEVEL_LABELS: Record<number, string> = {
  10: "DEBUG",
  20: "INFO",
  30: "WARNING",
  40: "ERROR",
  50: "CRITICAL",
};

export function getPrefectLogPageSize(): number {
  return LOG_PAGE_SIZE;
}

export function logLevelLabel(level: number): string {
  return LOG_LEVEL_LABELS[level] ?? `L${level}`;
}

export function mergePrefectLogs(
  existing: PrefectLogEntry[],
  incoming: PrefectLogEntry[],
): PrefectLogEntry[] {
  if (incoming.length === 0) {
    return existing;
  }

  const seen = new Set(existing.map((entry) => entry.id));
  const appended = incoming.filter((entry) => {
    if (seen.has(entry.id)) {
      return false;
    }
    seen.add(entry.id);
    return true;
  });

  return appended.length > 0 ? [...existing, ...appended] : existing;
}

export function buildPrefectLogFilterBody(
  flowRunIds: string[],
  offset = 0,
  limit = LOG_PAGE_SIZE,
) {
  return {
    offset,
    logs: {
      flow_run_id: { any_: flowRunIds },
    },
    sort: "TIMESTAMP_ASC",
    limit,
  };
}

export function buildPrefectLogStreamFilterMessage(
  flowRunIds: string[],
  now = new Date(),
) {
  return {
    type: "filter",
    filter: {
      flow_run_id: { any_: flowRunIds },
      timestamp: {
        after_: new Date(now.getTime() - LOG_STREAM_LOOKBACK_MS).toISOString(),
        before_: new Date(now.getTime() + LOG_STREAM_LOOKAHEAD_MS).toISOString(),
      },
    },
  };
}

async function fetchPrefectLogPage(
  flowRunIds: string[],
  offset = 0,
  limit = LOG_PAGE_SIZE,
): Promise<PrefectLogEntry[]> {
  if (flowRunIds.length === 0) return [];

  const res = await fetch("/prefect/logs/filter", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(buildPrefectLogFilterBody(flowRunIds, offset, limit)),
  });
  if (!res.ok) return [];
  return res.json();
}

export async function fetchIncrementalPrefectLogs(
  flowRunIds: string[],
  existing: PrefectLogEntry[],
  {
    limit = LOG_PAGE_SIZE,
    offset = existing.length,
  }: {
    limit?: number;
    offset?: number;
  } = {},
): Promise<PrefectLogEntry[]> {
  let merged = existing;
  let nextOffset = offset;

  while (true) {
    const nextPage = await fetchPrefectLogPage(flowRunIds, nextOffset, limit);
    if (nextPage.length === 0) {
      break;
    }

    merged = mergePrefectLogs(merged, nextPage);
    nextOffset += nextPage.length;

    if (nextPage.length < limit) {
      break;
    }
  }

  return merged;
}
