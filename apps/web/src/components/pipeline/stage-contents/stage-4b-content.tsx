import { RBPartitionCard } from "@/components/stages/parametric-id/rb-partition-card";
import { SensitivityAnalysisTable } from "@/components/stages/parametric-id/sensitivity-analysis-table";
import type { Stage4bData } from "@causal-ssm/api-types";

export default function Stage4bContent({ data }: { data: Stage4bData }) {
  const pid = data.parametric_id;

  return (
    <div className="space-y-4">
      {data.rb_partition && <RBPartitionCard partition={data.rb_partition} />}
      {pid.sensitivity_analysis && (
        <SensitivityAnalysisTable result={pid.sensitivity_analysis} />
      )}
    </div>
  );
}
