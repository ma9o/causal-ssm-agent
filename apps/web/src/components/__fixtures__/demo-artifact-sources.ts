import baselineReportArtifact from "../../../../../data/DEMO/fixture/artifacts/baseline_report.json";
import causalDesignArtifact from "../../../../../data/DEMO/fixture/artifacts/causal_design.json";
import latentStructureArtifact from "../../../../../data/DEMO/fixture/artifacts/latent_structure.json";
import measurementStructureArtifact from "../../../../../data/DEMO/fixture/artifacts/measurement_structure.json";
import measurementsArtifact from "../../../../../data/DEMO/fixture/artifacts/measurements.json";
import posteriorArtifact from "../../../../../data/DEMO/fixture/artifacts/posterior.json";
import rawDataArtifact from "../../../../../data/DEMO/fixture/artifacts/raw_data.json";
import statisticalModelSpecArtifact from "../../../../../data/DEMO/fixture/artifacts/statistical_model_spec.json";
import structuralPlanArtifact from "../../../../../data/DEMO/fixture/artifacts/structural_plan.json";
import validationReportArtifact from "../../../../../data/DEMO/fixture/artifacts/validation_report.json";

export const demoArtifactSources = {
  raw_data: rawDataArtifact,
  latent_structure: latentStructureArtifact,
  measurement_structure: measurementStructureArtifact,
  causal_design: causalDesignArtifact.causal_design,
  structural_plan: structuralPlanArtifact.structural_plan,
  measurements: measurementsArtifact,
  validation_report: validationReportArtifact,
  statistical_model_spec: statisticalModelSpecArtifact,
  posterior: posteriorArtifact,
  baseline_report: baselineReportArtifact,
} as const;
