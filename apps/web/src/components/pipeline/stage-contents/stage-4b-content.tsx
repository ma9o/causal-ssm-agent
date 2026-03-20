import { InferenceStructureCard } from "@/components/stages/parametric-id/inference-structure-card";
import { SensitivityAnalysisTable } from "@/components/stages/parametric-id/sensitivity-analysis-table";
import { TRuleCard } from "@/components/stages/parametric-id/t-rule-card";
import type { Stage4bData } from "@causal-ssm/api-types";

export default function Stage4bContent({ data }: { data: Stage4bData }) {
  const pid = data.parametric_id;

  return (
    <div className="space-y-4">
      {pid.t_rule && <TRuleCard tRule={pid.t_rule} />}
      {data.inference_structure && (
        <InferenceStructureCard inferenceStructure={data.inference_structure} />
      )}
      {pid.sensitivity_analysis && <SensitivityAnalysisTable result={pid.sensitivity_analysis} />}
    </div>
  );
}
