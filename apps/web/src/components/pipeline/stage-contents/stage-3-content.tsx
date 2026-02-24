import { IndicatorHealthTable } from "@/components/stages/validation/indicator-health-table";
import { HardGateAlert } from "@/components/ui/custom/hard-gate-alert";
import type { Stage3Data } from "@causal-ssm/api-types";

export default function Stage3Content({ data }: { data: Stage3Data }) {
  const report = data.validation_report;

  return (
    <div className="space-y-4">
      {!report.is_valid && (
        <HardGateAlert
          title="Data Validation Failed"
          explanation="The extracted data failed validation checks."
          suggestion="Check data source quality, add more indicators, or broaden measurement criteria."
        />
      )}
      {report.per_indicator_health.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold">Indicator Health</h3>
          <IndicatorHealthTable rows={report.per_indicator_health} />
        </div>
      )}
    </div>
  );
}
