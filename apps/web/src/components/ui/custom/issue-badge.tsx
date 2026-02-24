import { Badge } from "@/components/ui/badge";
import type { ValidationSeverity } from "@causal-ssm/api-types";

const severityVariant: Record<ValidationSeverity, "destructive" | "warning" | "secondary"> = {
  error: "destructive",
  warning: "warning",
  info: "secondary",
};

export function IssueBadge({ severity }: { severity: ValidationSeverity }) {
  return <Badge variant={severityVariant[severity]}>{severity}</Badge>;
}
