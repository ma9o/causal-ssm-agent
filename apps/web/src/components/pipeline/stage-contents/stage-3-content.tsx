import { IndicatorHealthTable } from "@/components/stages/validation/indicator-health-table";
import type { Stage3Data } from "@nof1-causal-lab/api-types";

export default function Stage3Content({ data }: { data: Stage3Data }) {
  const indicators = data.indicators ?? {};

  return (
    <div className="space-y-4">
      {Object.keys(indicators).length > 0 && <IndicatorHealthTable audits={indicators} />}
    </div>
  );
}
