import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import type { TRuleResult } from "@causal-ssm/api-types";

interface TRuleCardProps {
  tRule: TRuleResult;
}

export function TRuleCard({ tRule }: TRuleCardProps) {
  const paramCountEntries = Object.entries(tRule.param_counts);

  return (
    <Card>
      <CardContent className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2 py-3 text-sm">
        <span className="inline-flex items-center gap-1.5 font-medium">
          T-Rule
          <StatTooltip explanation="The t-rule checks if there are enough moment conditions (data constraints) to uniquely identify all free parameters. Requires moments ≥ free params." />
          <Badge variant={tRule.satisfies ? "success" : "destructive"} className="ml-0.5">
            {tRule.satisfies ? "Pass" : "Fail"}
          </Badge>
        </span>

        <span className="inline-flex items-center gap-1 text-muted-foreground">
          <span>Free params:</span>
          <span className="tabular-nums text-foreground">{tRule.n_free_params}</span>
          <StatTooltip explanation="The number of scalar parameters the model needs to estimate — each unknown coefficient, variance, or autoregressive term counts as one." />
          <span className="mx-0.5">≤</span>
          <span>Moments:</span>
          <span className="tabular-nums text-foreground">{tRule.n_moments}</span>
          <StatTooltip explanation="The number of independent equations the data provides — derived from variances and covariances of observed variables across time points." />
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
