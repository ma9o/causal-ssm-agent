import type { Stage5aData, Stage5bData } from "@causal-ssm/api-types";
import stage5aFixture from "../../../../../data/DEMO_HEALTH/run/stage-5a.json";
import stage5bFixture from "../../../../../data/DEMO_HEALTH/run/stage-5b.json";
import stage5bAuxGibbsFixture from "../../../../../data/DEMO_HEALTH/run/stage-5b-aux-gibbs.json";

export const stage5a = stage5aFixture as Stage5aData;
export const stage5b = stage5bFixture as Stage5bData;
export const stage5bAuxGibbs = stage5bAuxGibbsFixture as Stage5bData;
