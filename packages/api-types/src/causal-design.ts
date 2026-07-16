import type {
  CausalEdge,
  KnownInput,
  LatentStructure,
  MeasurementStructure,
} from "./generated/models";

export interface IdentifiedTreatmentStatus {
  method: string;
  estimand: string;
  marginalized_confounders: string[];
  instruments: string[];
}

export interface NonIdentifiableTreatmentStatus {
  confounders: string[];
  notes?: string | null;
}

export interface IdentifiabilityStatus {
  identifiable_treatments: Record<string, IdentifiedTreatmentStatus>;
  non_identifiable_treatments: Record<string, NonIdentifiableTreatmentStatus>;
}

export interface InducedDependency {
  between: [string, string];
  kind: "innovation_correlation" | "initial_state_correlation";
  source_confounders: string[];
}

export interface EstimationSpec {
  state_order: string[];
  edges: CausalEdge[];
  induced_dependencies: InducedDependency[];
  known_inputs: KnownInput[];
}

export interface CausalDesign {
  latent: LatentStructure;
  measurement: MeasurementStructure;
  identifiability?: IdentifiabilityStatus | null;
  estimation?: EstimationSpec | null;
}
