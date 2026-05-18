import { InferenceStructureCard } from "@/components/stages/parametric-id/inference-structure-card";
import { MapGeometryPanel } from "@/components/stages/parametric-id/map-geometry-panel";
import { SensitivityAnalysisTable } from "@/components/stages/parametric-id/sensitivity-analysis-table";
import type { Stage4bData } from "@nof1-causal-lab/api-types";

export default function Stage4bContent({ data }: { data: Stage4bData }) {
  const pid = data.parametric_id;

  return (
    <div className="space-y-4">
      {data.inference_structure && (
        <InferenceStructureCard inferenceStructure={data.inference_structure} />
      )}
      {pid.sensitivity_analysis && <SensitivityAnalysisTable result={pid.sensitivity_analysis} />}
      {pid.map_geometry && <MapGeometryPanel result={pid.map_geometry} />}
    </div>
  );
}
