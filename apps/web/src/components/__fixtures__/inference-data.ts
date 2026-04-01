import type { Stage5aData, Stage5bData } from "@causal-ssm/api-types";
import stage5aFixture from "../../../../../data/DOCTOLIB/run/stage-5a.json";
import stage5bFixture from "../../../../../data/DOCTOLIB/run/stage-5b.json";
import stage5bNutsdaFixture from "../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";

export const stage5a = stage5aFixture as Stage5aData;
export const stage5b = stage5bFixture as Stage5bData;
export const stage5bNutsda = stage5bNutsdaFixture as Stage5bData;
