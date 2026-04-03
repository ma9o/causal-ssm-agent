"use client";

import { Skeleton } from "@/components/ui/skeleton";
import type { AccessibleWorkspaceEntry, AccessibleWorkspaceList, AccessibleWorkspaceSource } from "@/lib/server/workspace-ownership";
import { cn } from "@/lib/utils";
import { Database, FolderClock, FolderKanban, FolderOpen } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";

const MODE_DESCRIPTIONS: Record<AccessibleWorkspaceList["mode"], string> = {
  anonymous: "Anonymous mode keeps private workspace access inside this browser session.",
  user: "User mode persists private workspace access against your OpenRouter account.",
  local: "Local mode treats every workspace under ./data as owned.",
};

type SectionMeta = {
  title: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
};

const SECTION_META: Record<AccessibleWorkspaceSource, SectionMeta> = {
  local: { title: "Local data", description: "Every workspace under ./data is owned in local mode.", icon: Database },
  session: { title: "Browser session", description: "Private workspaces in this browser session.", icon: FolderClock },
  shared: { title: "Shared fixtures", description: "Shared demo workspaces available to every user.", icon: FolderOpen },
  user: { title: "OpenRouter account", description: "Workspaces persisted against your OpenRouter user.", icon: FolderKanban },
};

const SOURCE_ORDER: AccessibleWorkspaceSource[] = ["user", "local", "session", "shared"];

function groupBySources(workspaces: AccessibleWorkspaceEntry[]): Map<AccessibleWorkspaceSource, AccessibleWorkspaceEntry[]> {
  const groups = new Map<AccessibleWorkspaceSource, AccessibleWorkspaceEntry[]>();
  for (const workspace of workspaces) {
    const list = groups.get(workspace.source);
    if (list) {
      list.push(workspace);
    } else {
      groups.set(workspace.source, [workspace]);
    }
  }
  return groups;
}

export function AccessibleWorkspacesRail({
  currentWorkspaceId,
  data,
  error,
  isLoading,
}: {
  currentWorkspaceId: string;
  data?: AccessibleWorkspaceList;
  error?: string | null;
  isLoading: boolean;
}) {
  const grouped = useMemo(
    () => (data ? groupBySources(data.workspaces) : null),
    [data],
  );

  return (
    <aside className="border-t border-border/70 bg-muted/10 xl:border-t-0 xl:border-l">
      <div className="px-4 py-6 sm:px-6 xl:sticky xl:top-16 xl:max-h-[calc(100vh-4rem)] xl:overflow-y-auto xl:px-5">
        <div className="space-y-5">
          <div className="space-y-2">
            <p className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">
              Accessible Workspaces
            </p>
            <p className="text-sm text-muted-foreground">
              {data ? MODE_DESCRIPTIONS[data.mode] : "Loading workspace ownership..."}
            </p>
          </div>

          {isLoading && (
            <div className="space-y-3">
              <Skeleton className="h-4 w-32" />
              <Skeleton className="h-14 w-full" />
              <Skeleton className="h-14 w-full" />
              <Skeleton className="h-4 w-28" />
              <Skeleton className="h-14 w-full" />
            </div>
          )}

          {!isLoading && error && (
            <div className="rounded-2xl border border-dashed border-border/80 bg-background/70 p-4 text-sm text-muted-foreground">
              {error}
            </div>
          )}

          {!isLoading && !error && data?.workspaces.length === 0 && (
            <div className="rounded-2xl border border-dashed border-border/80 bg-background/70 p-4 text-sm text-muted-foreground">
              No accessible workspaces were found for this mode yet.
            </div>
          )}

          {!isLoading &&
            !error &&
            grouped &&
            SOURCE_ORDER.filter((source) => grouped.has(source)).map((source) => {
              const meta = SECTION_META[source];
              const Icon = meta.icon;
              const workspaces = grouped.get(source)!;
              return (
                <section key={source} className="space-y-2.5">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2 text-sm font-medium text-foreground">
                      <Icon className="size-4" />
                      <span>{meta.title}</span>
                    </div>
                    <p className="text-xs leading-5 text-muted-foreground">
                      {meta.description}
                    </p>
                  </div>
                  <div className="space-y-2">
                    {workspaces.map((workspace) => {
                      const isCurrent = workspace.workspaceId === currentWorkspaceId;
                      return (
                        <Link
                          key={`${source}:${workspace.workspaceId}`}
                          href={workspace.href}
                          className={cn(
                            "block rounded-2xl border bg-background/85 px-3 py-3 transition-colors hover:bg-background",
                            isCurrent && "border-foreground/50 bg-background shadow-sm",
                          )}
                        >
                          <div className="flex items-center justify-between gap-3">
                            <p className="font-mono text-xs font-semibold tracking-[0.16em] text-foreground">
                              {workspace.workspaceId}
                            </p>
                            {isCurrent && (
                              <span className="rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                                Current
                              </span>
                            )}
                          </div>
                          <p className="mt-1 text-sm leading-5 text-muted-foreground">
                            {workspace.question ?? "Question not available yet."}
                          </p>
                        </Link>
                      );
                    })}
                  </div>
                </section>
              );
            })}
        </div>
      </div>
    </aside>
  );
}
