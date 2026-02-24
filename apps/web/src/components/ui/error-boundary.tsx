"use client";

import { type ReactNode } from "react";
import { ErrorBoundary as ReactErrorBoundary, type FallbackProps } from "react-error-boundary";

function DefaultFallback({ error, resetErrorBoundary }: FallbackProps) {
  const message = error instanceof Error ? error.message : String(error);
  return (
    <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4">
      <p className="text-sm font-medium text-destructive">Something went wrong</p>
      <p className="mt-1 text-xs text-muted-foreground">{message}</p>
      <button
        type="button"
        className="mt-3 text-xs font-medium text-primary underline underline-offset-2 hover:no-underline"
        onClick={resetErrorBoundary}
      >
        Try again
      </button>
    </div>
  );
}

export function ErrorBoundary({
  children,
  fallback,
  resetKeys,
}: {
  children: ReactNode;
  fallback?: ReactNode;
  resetKeys?: unknown[];
}) {
  return (
    <ReactErrorBoundary
      FallbackComponent={fallback ? () => <>{fallback}</> : DefaultFallback}
      resetKeys={resetKeys}
    >
      {children}
    </ReactErrorBoundary>
  );
}
