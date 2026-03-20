"use client";

import { Table2, Download } from "lucide-react";
import { useParams } from "next/navigation";

export function ExploreDataframeButton({ stage }: { stage: string }) {
  const { userId } = useParams<{ userId: string }>();

  return (
    <div className="flex items-center gap-1">
      <a
        href={`/explore/${encodeURIComponent(userId)}/${encodeURIComponent(stage)}`}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex items-center gap-1.5 rounded-md border border-muted bg-muted/50 px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted"
      >
        <Table2 className="h-3.5 w-3.5" />
        Explore full dataset
      </a>
      <a
        href={`/api/results/${userId}/${stage}/dataframe`}
        download
        className="inline-flex items-center gap-1 rounded-md border border-muted bg-muted/50 px-2 py-1.5 text-xs text-muted-foreground transition-colors hover:bg-muted"
        title="Download parquet"
      >
        <Download className="h-3.5 w-3.5" />
      </a>
    </div>
  );
}
