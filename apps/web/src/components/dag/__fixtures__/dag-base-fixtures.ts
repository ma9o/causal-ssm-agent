import type { LatentStructureData, MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import latentFixture from "../../__fixtures__/demo-run/latent_structure.json";
import measurementFixture from "../../__fixtures__/demo-run/measurement_structure.json";

const latentStructure = latentFixture as unknown as LatentStructureData;
const measurementStructure = measurementFixture as unknown as MeasurementStructureViewData;

export const constructs = latentStructure.latent_structure.constructs;
export const edges = latentStructure.latent_structure.edges;
const spec = measurementStructure.causal_design;
export const indicators = spec.measurement.indicators;
