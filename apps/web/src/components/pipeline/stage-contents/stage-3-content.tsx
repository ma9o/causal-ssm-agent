import { IndicatorHealthTable } from "@/components/stages/validation/indicator-health-table";
import type { Stage3Data, ValidationIssue } from "@causal-ssm/api-types";

/** Collect all per-indicator issues into a flat list. */
function collectAllIssues(data: Stage3Data): ValidationIssue[] {
  const issues: ValidationIssue[] = [];

  for (const [name, audit] of Object.entries(data.indicators ?? {})) {
    for (const issue of audit?.validation?.issues ?? []) {
      issues.push({ ...issue, indicator: issue.indicator ?? name });
    }
  }

  for (const issue of data.dataset_issues ?? []) {
    issues.push(issue);
  }

  return issues;
}

/** Group issues by issue_type and build a categorized prompt for stage 1b. */
export function buildFixPrompt(data: Stage3Data): string {
  const issues = collectAllIssues(data).filter((i) => i.severity === "error");
  if (issues.length === 0) return "";

  const byType = new Map<string, ValidationIssue[]>();
  for (const issue of issues) {
    const existing = byType.get(issue.issue_type) ?? [];
    existing.push(issue);
    byType.set(issue.issue_type, existing);
  }

  const lines: string[] = [
    "Stage 3 (Validation) failed with the following measurement issues:\n",
  ];

  for (const [type, group] of byType) {
    const label = type.replaceAll("_", " ");
    lines.push(`**${label}** (${group.length}):`);
    for (const issue of group) {
      const prefix = issue.indicator ? `${issue.indicator}: ` : "";
      lines.push(`- ${prefix}${issue.message}`);
    }
    lines.push("");
  }

  lines.push(
    "Please revise the measurement model to fix these issues. Consider removing indicators with insufficient data, adjusting measurement definitions, or merging similar indicators.",
  );

  return lines.join("\n");
}

export default function Stage3Content({ data }: { data: Stage3Data }) {
  const indicators = data.indicators ?? {};
  const datasetIssues = data.dataset_issues ?? [];

  return (
    <div className="space-y-4">
      {Object.keys(indicators).length > 0 && (
        <IndicatorHealthTable audits={indicators} />
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
