import type { LatentStructureData, MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import { demoLatentStructure, demoMeasurementStructure } from "../../__fixtures__/demo-artifacts";

const latentStructure = demoLatentStructure as LatentStructureData;
const measurementStructure = demoMeasurementStructure as MeasurementStructureViewData;

export const constructs = latentStructure.latent_structure.constructs;
export const edges = latentStructure.latent_structure.edges;
const spec = measurementStructure.causal_design;
export const indicators = spec.measurement.indicators;
