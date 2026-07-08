import type { PosteriorData } from "@nof1-causal-lab/api-types";
import posteriorFixture from "./demo-run/posterior.json";

export const posterior = posteriorFixture as PosteriorData;
export const posteriorAuxKalmanMCMC = posteriorFixture as PosteriorData;
