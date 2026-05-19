export const SHARED_WORKSPACE_IDS = ["DEFAULT", "DEMO", "GOLDEN", "SMALLGOLDEN"] as const;

export type SharedWorkspaceId = (typeof SHARED_WORKSPACE_IDS)[number];

const SHARED_WORKSPACE_ALIASES = {
  DEMO_HEALTH: "DEMO",
} as const satisfies Record<string, SharedWorkspaceId>;

export function resolveSharedWorkspaceId(workspaceId: string): SharedWorkspaceId | null {
  const normalized = workspaceId.toUpperCase();
  if (SHARED_WORKSPACE_IDS.includes(normalized as SharedWorkspaceId)) {
    return normalized as SharedWorkspaceId;
  }
  return SHARED_WORKSPACE_ALIASES[normalized as keyof typeof SHARED_WORKSPACE_ALIASES] ?? null;
}

export function isSharedWorkspaceId(workspaceId: string): boolean {
  return resolveSharedWorkspaceId(workspaceId) !== null;
}
