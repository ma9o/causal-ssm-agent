"use client";

import type {
  AccessibleWorkspaceEntry,
  AccessibleWorkspaceList,
} from "@/lib/server/workspace-ownership";
import { Skeleton } from "@/components/ui/skeleton";
import Link from "next/link";

export function AccessibleWorkspacesRail({
  data,
  error,
  isLoading,
}: {
  data?: AccessibleWorkspaceList;
  error?: string | null;
  isLoading: boolean;
}) {
  const workspaces = data?.workspaces ?? [];

  return (
    <div className="w-full max-w-2xl mx-auto space-y-3">
      {isLoading && (
        <>
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </>
      )}

      {!isLoading && error && (
        <p className="text-sm text-muted-foreground text-center py-8">
          {error}
        </p>
      )}

      {!isLoading && !error && workspaces.length === 0 && (
        <p className="text-sm text-muted-foreground text-center py-8">
          No workspaces yet. Start an analysis to create one.
        </p>
      )}

      {!isLoading &&
        !error &&
        workspaces.map((workspace) => (
          <WorkspaceCard key={workspace.workspaceId} workspace={workspace} />
        ))}
    </div>
  );
}

function WorkspaceCard({ workspace }: { workspace: AccessibleWorkspaceEntry }) {
  return (
    <Link
      href={workspace.href}
      className="block rounded-lg border bg-card px-4 py-3 shadow-sm transition-colors hover:bg-accent/50"
    >
      <p className="font-mono text-xs font-semibold tracking-wider text-muted-foreground">
        {workspace.workspaceId}
      </p>
      <p className="mt-1 text-sm leading-snug text-foreground">
        {workspace.question ?? "Question not available yet."}
      </p>
    </Link>
  );
}

function SkeletonCard() {
  return (
    <div className="rounded-lg border bg-card px-4 py-3 shadow-sm space-y-2">
      <Skeleton className="h-3 w-24" />
      <Skeleton className="h-4 w-full" />
    </div>
  );
}
