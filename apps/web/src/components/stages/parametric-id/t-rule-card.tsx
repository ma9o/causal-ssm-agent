import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import type { TRuleResult } from "@causal-ssm/api-types";

export function TRuleCard({
  tRule,
}: {
  tRule: TRuleResult;
}) {
  const paramCountEntries = Object.entries(tRule.param_counts ?? {});
  const statusLabel = tRule.satisfies ? "Pass" : "Warning";
  const statusVariant = tRule.satisfies ? "success" : "warning";

  return (
    <Card>
      <CardContent className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 py-3 text-sm">
        <span className="inline-flex items-center gap-1.5 font-medium">
          T-Rule
          <StatTooltip explanation="Conservative counting screen comparing free parameters against a lower bound on available observed moment conditions. Passing does not guarantee identifiability; failing is warning-only." />
          <Badge variant={statusVariant} className="ml-0.5">
            {statusLabel}
          </Badge>
        </span>

        <span className="inline-flex items-center gap-1 text-muted-foreground">
          <span>Free params:</span>
          <span className="tabular-nums text-foreground">{tRule.n_free_params}</span>
          <StatTooltip explanation="The number of scalar parameters the model needs to estimate, counted from the canonical site registry." />
          <span className="mx-0.5">≤</span>
          <span>Lower-bound moments:</span>
          <span className="tabular-nums text-foreground">{tRule.n_moments}</span>
          <StatTooltip explanation="A conservative lower bound on the number of observed moment conditions contributed by means, contemporaneous covariance, and lagged autocovariance." />
        </span>

        {paramCountEntries.length > 0 &&
          paramCountEntries.map(([key, count]) => (
            <Badge key={key} variant="secondary">
              {key}: {count}
            </Badge>
          ))}
      </CardContent>
    </Card>
  );
}
