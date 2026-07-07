import type { Stage1aData, Stage1bData } from "@nof1-causal-lab/api-types";
import stage1aFixture from "../../__fixtures__/demo-run/stage-1a.json";
import stage1bFixture from "../../__fixtures__/demo-run/stage-1b.json";

const stage1a = stage1aFixture as unknown as Stage1aData;
const stage1b = stage1bFixture as unknown as Stage1bData;

export const constructs = stage1a.latent_model.constructs;
export const edges = stage1a.latent_model.edges;
const spec = stage1b.causal_spec;
export const indicators = spec.measurement.indicators;
