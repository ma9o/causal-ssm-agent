export const SHARED_WORKSPACE_IDS = [
  "DEFAULT",
  "DEMO",
  "DEMO_HEALTH",
  "GOLDEN",
  "SMALLGOLDEN",
] as const;

export function isSharedWorkspaceId(workspaceId: string): boolean {
  return SHARED_WORKSPACE_IDS.includes(
    workspaceId.toUpperCase() as (typeof SHARED_WORKSPACE_IDS)[number],
  );
}
