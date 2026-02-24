import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils/cn";
import { AlertCircle } from "lucide-react";

interface RetryEntry {
  attempt: number;
  failed_params: string[];
  feedback: string;
}

interface RetryIndicatorProps {
  retries: RetryEntry[];
}

export function RetryIndicator({ retries }: RetryIndicatorProps) {
  if (retries.length === 0) return null;

  return (
    <div className="space-y-0">
      {retries.map((retry, i) => (
        <div
          key={`retry-${
            // biome-ignore lint/suspicious/noArrayIndexKey: stable ordered list
            i
          }`}
          className="relative flex gap-4 pb-6 last:pb-0"
        >
          {/* Vertical line */}
          <div className="flex flex-col items-center">
            <div
              className={cn(
                "flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2",
                "border-warning bg-warning/15 text-warning-foreground",
              )}
            >
              <AlertCircle className="h-4 w-4" />
            </div>
            {i < retries.length - 1 && <div className="w-px flex-1 bg-border" />}
          </div>

          {/* Content */}
          <div className="flex-1 pt-1">
            <p className="text-sm font-medium">Attempt {retry.attempt}</p>
            <div className="mt-1 flex flex-wrap gap-1">
              {retry.failed_params.map((param) => (
                <Badge key={param} variant="destructive" className="text-xs">
                  {param}
                </Badge>
              ))}
            </div>
            <p className="mt-2 text-sm text-muted-foreground">{retry.feedback}</p>
          </div>
        </div>
      ))}
    </div>
  );
}
