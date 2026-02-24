import { Card, CardContent } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import type { RBPartitionResult } from "@causal-ssm/api-types";

interface RBPartitionCardProps {
  partition: RBPartitionResult;
}

export function RBPartitionCard({ partition }: RBPartitionCardProps) {
  const kalman = partition.latent_variables.filter((v) => v.method === "kalman");
  const particle = partition.latent_variables.filter((v) => v.method === "particle");

  return (
    <Card>
      <CardContent className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 py-3 text-sm">
        <span className="inline-flex items-center gap-1.5 font-medium">
          Rao-Blackwellization
        </span>

        {/* Proportion bar */}
        <div className="min-w-24 max-w-64 flex-1">
          <div className="flex h-2.5 overflow-hidden rounded-full bg-muted">
            {kalman.length > 0 && (
              <div
                className="h-full bg-success"
                style={{ flex: kalman.length }}
              />
            )}
            {particle.length > 0 && (
              <div
                className="h-full bg-warning"
                style={{ flex: particle.length }}
              />
            )}
          </div>
        </div>

        {/* Counts */}
        <span className="inline-flex items-center gap-1.5 text-muted-foreground">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-success" />
          <span className="tabular-nums text-foreground">{kalman.length}</span>
          <span>Marginalized</span>
          <span className="mx-1 text-muted-foreground/40">|</span>
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-warning" />
          <span className="tabular-nums text-foreground">{particle.length}</span>
          <span>Require Sampling</span>
        </span>
      </CardContent>
    </Card>
  );
}
