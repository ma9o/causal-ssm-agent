import type { ArtifactViewId } from "@nof1-causal-lab/api-types";
import rawDataArtifact from "../../../../../data/DEMO/store/raw_data/v1/profile.json";
import latentStructureArtifact from "../../../../../data/DEMO/store/latent_structure/v1/latent-structure.json";
import measurementStructureArtifact from "../../../../../data/DEMO/store/measurement_structure/v1/measurement_structure.json";
import causalDesignArtifact from "../../../../../data/DEMO/store/causal_design/v1/causal_design.json";
import measurementsArtifact from "../../../../../data/DEMO/store/measurements/v1/measurements.json";
import validationReportArtifact from "../../../../../data/DEMO/store/validation_report/v1/validation_report.json";
import baselineReportArtifact from "./demo-run/baseline_report.json";
import posteriorArtifact from "./demo-run/posterior.json";
import statisticalModelSpecArtifact from "./demo-run/statistical_model_spec.json";

export const demoArtifactSources = {
  store: {
    raw_data: rawDataArtifact,
    latent_structure: latentStructureArtifact,
    measurement_structure: measurementStructureArtifact,
    causal_design: causalDesignArtifact.causal_design,
    measurements: measurementsArtifact,
    validation_report: validationReportArtifact,
  },
  temporaryDemoRun: {
    statistical_model_spec: statisticalModelSpecArtifact,
    posterior: posteriorArtifact,
    baseline_report: baselineReportArtifact,
  },
} as const;

// These are the canonical DEMO store files that should replace temporaryDemoRun
// as soon as the episode is materialized through the current artifact backend.
export const temporaryDemoRunArtifactTargets = {
  statistical_model_spec: "data/DEMO/store/statistical_model_spec/v1/statistical_model_spec.json",
  posterior: "data/DEMO/store/posterior/v1/diagnostics.json",
  baseline_report: "data/DEMO/store/baseline_report/v1/baseline_report.json",
} as const satisfies Partial<Record<ArtifactViewId, string>>;
