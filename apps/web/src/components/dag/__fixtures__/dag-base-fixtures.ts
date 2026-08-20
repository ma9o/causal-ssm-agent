import type { LatentStructureData, MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import { demoLatentStructure, demoMeasurementStructure } from "../../__fixtures__/demo-artifacts";

const latentStructure = demoLatentStructure as LatentStructureData;
const measurementStructure = demoMeasurementStructure as MeasurementStructureViewData;
export const design = measurementStructure.causal_design;
export const structuralPlan = measurementStructure.structural_plan;
export const constructs = latentStructure.latent_structure.constructs;
export const edges = latentStructure.latent_structure.edges;
export const indicators = design.measurement.indicators;
export const knownInputs = design.known_inputs;
