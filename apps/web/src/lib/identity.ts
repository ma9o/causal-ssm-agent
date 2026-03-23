// Persistent workspace identity — survives sign-out (only API key is cleared)

export type WorkspaceIdentity = {
  workspaceId: string;
  accessCode: string;
  kind: "anonymous" | "openrouter";
};

const IDENTITY_KEY = "workspace_identity";

export function getIdentity(): WorkspaceIdentity | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(IDENTITY_KEY);
    if (!raw) return null;
    const identity = JSON.parse(raw) as Partial<WorkspaceIdentity>;
    if (
      typeof identity?.workspaceId !== "string" ||
      typeof identity?.accessCode !== "string" ||
      (identity?.kind !== "anonymous" && identity?.kind !== "openrouter")
    ) {
      return null;
    }
    return identity as WorkspaceIdentity;
  } catch {
    return null;
  }
}

export function setIdentity(identity: WorkspaceIdentity): void {
  localStorage.setItem(IDENTITY_KEY, JSON.stringify(identity));
}

export function clearIdentity(): void {
  localStorage.removeItem(IDENTITY_KEY);
}
