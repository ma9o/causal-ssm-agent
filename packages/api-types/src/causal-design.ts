import type {
  KnownInput,
  LatentStructure,
  MeasurementStructure,
  ScientificOnlyConstruct,
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

export interface CausalDesign {
  latent: LatentStructure;
  measurement: MeasurementStructure;
  identifiability?: IdentifiabilityStatus | null;
  known_inputs: KnownInput[];
  scientific_only_constructs: ScientificOnlyConstruct[];
}
