import { IndicatorHealthTable } from "@/components/analysis-widgets/validation-report/indicator-health-table";
import type { ValidationReportData } from "@nof1-causal-lab/api-types";

export default function ValidationReportView({ data }: { data: ValidationReportData }) {
  const indicators = data.indicators ?? {};

  return (
    <div className="space-y-4">
      {Object.keys(indicators).length > 0 && <IndicatorHealthTable audits={indicators} />}
    </div>
  );
}
