import { IndicatorHealthTable } from "@/components/stages/validation/indicator-health-table";
import type { Stage3Data } from "@causal-ssm/api-types";

export default function Stage3Content({ data }: { data: Stage3Data }) {
  const indicators = data.indicators ?? {};
  const datasetIssues = data.dataset_issues ?? [];

  return (
    <div className="space-y-4">
      {Object.keys(indicators).length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Indicator Audits</h3>
          <IndicatorHealthTable audits={indicators} />
        </div>
      )}
      {datasetIssues.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-semibold">Dataset Issues</h3>
          <ul className="space-y-1 text-sm text-muted-foreground">
            {datasetIssues.map((issue, index) => (
              <li key={`${issue.issue_type}-${index}`}>
                {issue.indicator ? `${issue.indicator}: ` : ""}
                {issue.message}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
