export const ACCESS_CODE_CHARSET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ";
export const ACCESS_CODE_LENGTH = 24;
export const SHARED_WORKSPACE_ACCESS_CODE = "test";
export const SHARED_WORKSPACE_IDS = [
  "DEFAULT",
  "DOCTOLIB",
  "MEDICAL_SEMANTICS",
  "GOLDEN",
  "SMALLGOLDEN",
] as const;

export function generateWorkspaceAccessCode(): string {
  const values = crypto.getRandomValues(new Uint8Array(ACCESS_CODE_LENGTH));
  return Array.from(values, (value) => ACCESS_CODE_CHARSET[value % ACCESS_CODE_CHARSET.length]).join("");
}

export function isSharedWorkspaceId(workspaceId: string): boolean {
  return SHARED_WORKSPACE_IDS.includes(workspaceId.toUpperCase() as (typeof SHARED_WORKSPACE_IDS)[number]);
}

export function getSharedWorkspaceAccessCode(workspaceId: string): string | null {
  return isSharedWorkspaceId(workspaceId) ? SHARED_WORKSPACE_ACCESS_CODE : null;
}

export function formatResumeKey(workspaceId: string, accessCode: string): string {
  return `${workspaceId}.${accessCode}`;
}

export function parseResumeKey(value: string): { workspaceId: string; accessCode: string | null } | null {
  const trimmed = value.trim();
  if (!trimmed) {
    return null;
  }

  const delimiterIndex = trimmed.indexOf(".");
  if (delimiterIndex === -1) {
    return { workspaceId: trimmed, accessCode: null };
  }

  const workspaceId = trimmed.slice(0, delimiterIndex).trim();
  const accessCode = trimmed.slice(delimiterIndex + 1).trim();
  if (!workspaceId) {
    return null;
  }

  return { workspaceId, accessCode: accessCode || null };
}
