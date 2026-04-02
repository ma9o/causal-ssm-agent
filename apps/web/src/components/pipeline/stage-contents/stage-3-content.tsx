import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { IndicatorHealthTable } from "@/components/stages/validation/indicator-health-table";
import type { Stage3Data, ValidationIssue } from "@causal-ssm/api-types";
import { Wrench } from "lucide-react";

export type Stage3FixSeverity = "warning" | "error";

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

function collectActionableIssues(data: Stage3Data): ValidationIssue[] {
  return collectAllIssues(data).filter((issue) => issue.severity !== "info");
}

export function getFixPromptData(data: Stage3Data): {
  prompt: string;
  highestSeverity: Stage3FixSeverity | null;
} {
  const issues = collectActionableIssues(data);
  if (issues.length === 0) {
    return { prompt: "", highestSeverity: null };
  }

  const highestSeverity: Stage3FixSeverity = issues.some((issue) => issue.severity === "error")
    ? "error"
    : "warning";

  const bySeverity = new Map<ValidationIssue["severity"], ValidationIssue[]>();
  for (const issue of issues) {
    const existing = bySeverity.get(issue.severity) ?? [];
    existing.push(issue);
    bySeverity.set(issue.severity, existing);
  }

  const lines: string[] = [
    "Stage 3 (Validation) surfaced the following measurement issues:\n",
  ];

  for (const severity of ["error", "warning"] as const) {
    const group = bySeverity.get(severity) ?? [];
    if (group.length === 0) continue;

    lines.push(`**${severity === "error" ? "Errors" : "Warnings"}** (${group.length}):`);

    const byType = new Map<string, ValidationIssue[]>();
    for (const issue of group) {
      const existing = byType.get(issue.issue_type) ?? [];
      existing.push(issue);
      byType.set(issue.issue_type, existing);
    }

    for (const [type, typedGroup] of byType) {
      const label = type.replaceAll("_", " ");
      lines.push(`- ${label}:`);
      for (const issue of typedGroup) {
        const prefix = issue.indicator ? `${issue.indicator}: ` : "";
        lines.push(`  - ${prefix}${issue.message}`);
      }
    }
    lines.push("");
  }

  lines.push(
    "Please revise the measurement model to fix these issues. Consider removing indicators with insufficient data, adjusting measurement definitions, or merging similar indicators.",
  );

  return {
    prompt: lines.join("\n"),
    highestSeverity,
  };
}

export function Stage3FixAction({
  data,
  onFix,
}: {
  data: Stage3Data;
  onFix: (prompt: string) => void;
}) {
  const { prompt, highestSeverity } = getFixPromptData(data);

  if (!prompt || !highestSeverity) {
    return null;
  }

  return (
    <Button
      type="button"
      onClick={() => onFix(prompt)}
      variant={highestSeverity === "error" ? "destructive" : "outline"}
      size="sm"
      className={cn(
        highestSeverity === "warning" &&
          "border-warning/50 bg-warning/15 text-warning-foreground hover:bg-warning/20 focus-visible:border-warning/40 focus-visible:ring-warning/20",
      )}
    >
      <Wrench className="h-3.5 w-3.5" />
      Propose fixes
    </Button>
  );
}

/** Group issues by issue_type and build a categorized prompt for stage 1b. */
export function buildFixPrompt(data: Stage3Data): string {
  return getFixPromptData(data).prompt;
}

export default function Stage3Content({ data }: { data: Stage3Data }) {
  const indicators = data.indicators ?? {};

  return (
    <div className="space-y-4">
      {Object.keys(indicators).length > 0 && (
        <IndicatorHealthTable audits={indicators} />
      )}
    </div>
  );
}
