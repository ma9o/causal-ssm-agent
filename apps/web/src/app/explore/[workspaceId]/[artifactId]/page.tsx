"use client";

import { useEffect, useRef, useState, use } from "react";

type PerspectiveViewerElement = HTMLElement & {
  load: (table: unknown) => Promise<void> | void;
};

export default function ExplorePage({
  params,
}: {
  params: Promise<{ workspaceId: string; artifactId: string }>;
}) {
  const { workspaceId, artifactId } = use(params);
  const containerRef = useRef<HTMLDivElement>(null);
  const [status, setStatus] = useState<"loading" | "ready" | "error">("loading");
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;

    async function init() {
      try {
        const resp = await fetch(`/api/artifacts/${workspaceId}/${artifactId}/dataframe`);
        if (!resp.ok) {
          throw new Error(
            resp.status === 404
              ? "No dataframe available. The pipeline may not have produced a parquet file for this artifact."
              : `Failed to fetch dataframe: ${resp.statusText}`,
          );
        }
        const buffer = await resp.arrayBuffer();
        if (cancelled) return;

        // 2. Parse parquet → JSON rows (pure JS, no WASM)
        const { parquetMetadata, parquetReadObjects } = await import("hyparquet");
        const { compressors } = await import("hyparquet-compressors");

        // Read schema to identify timestamp/date columns
        const metadata = parquetMetadata(buffer);
        const temporalCols = new Set<string>();
        for (const col of metadata.schema) {
          const ct = col.converted_type;
          if (
            ct === "TIMESTAMP_MILLIS" ||
            ct === "TIMESTAMP_MICROS" ||
            ct === "DATE" ||
            ct === "TIME_MILLIS" ||
            ct === "TIME_MICROS"
          ) {
            temporalCols.add(col.name);
          }
        }

        const raw = await parquetReadObjects({ file: buffer, compressors });

        // Convert BigInts: timestamps → Date objects, others → Number
        const rows = raw.map((row: Record<string, unknown>) => {
          const out: Record<string, unknown> = {};
          for (const [k, v] of Object.entries(row)) {
            if (typeof v !== "bigint") {
              out[k] = v;
            } else if (temporalCols.has(k)) {
              // Polars writes TIMESTAMP_MICROS by default
              out[k] = new Date(Number(v / BigInt(1000)));
            } else {
              out[k] = Number(v);
            }
          }
          return out;
        });
        if (cancelled || !containerRef.current) return;

        // 3. Initialize Perspective with WASM from public/
        const perspective = await import("@finos/perspective");
        perspective.init_server(fetch("/perspective/wasm/perspective-server.wasm"));

        const { init_client } = await import("@finos/perspective-viewer");
        await init_client(fetch("/perspective/wasm/perspective-viewer.wasm"));

        // 4. Register plugins (must be after viewer init)
        await import("@finos/perspective-viewer-datagrid");
        await import("@finos/perspective-viewer-d3fc");

        if (cancelled || !containerRef.current) return;

        // 5. Create worker, table, and viewer
        const worker = await perspective.worker();
        const table = worker.table(rows);

        const viewer = document.createElement("perspective-viewer") as PerspectiveViewerElement;
        viewer.setAttribute("theme", "Pro Dark");
        viewer.style.width = "100%";
        viewer.style.height = "100%";
        containerRef.current.innerHTML = "";
        containerRef.current.appendChild(viewer);
        await viewer.load(table);

        if (!cancelled) setStatus("ready");
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setStatus("error");
        }
      }
    }

    init();
    return () => {
      cancelled = true;
    };
  }, [workspaceId, artifactId]);

  const artifactLabel = artifactId.replace("-", " ").replace(/\b\w/g, (c) => c.toUpperCase());

  return (
    <div className="flex h-screen w-screen flex-col bg-neutral-950 text-neutral-200">
      <link rel="stylesheet" href="/perspective/pro-dark.css" precedence="default" />
      <header className="flex shrink-0 items-center justify-between border-b border-neutral-800 px-4 py-2">
        <div className="flex items-center gap-3">
          <h1 className="text-sm font-medium">{artifactLabel} - Full Dataset</h1>
          <span className="text-xs text-neutral-500">{workspaceId}</span>
        </div>
        {status === "loading" && (
          <span className="animate-pulse text-xs text-neutral-500">Loading dataframe...</span>
        )}
      </header>

      {status === "error" && (
        <div className="flex flex-1 items-center justify-center">
          <div className="max-w-md rounded-lg border border-red-900/50 bg-red-950/30 p-6 text-center">
            <p className="text-sm text-red-400">{error}</p>
          </div>
        </div>
      )}

      <div
        ref={containerRef}
        className="flex-1"
        style={{ visibility: status === "error" ? "hidden" : "visible" }}
      />
    </div>
  );
}
