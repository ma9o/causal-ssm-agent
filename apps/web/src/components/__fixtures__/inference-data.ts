import type { PosteriorData } from "@nof1-causal-lab/api-types";
import { demoPosterior } from "./demo-artifacts";

export const posterior = demoPosterior as PosteriorData;
export const posteriorAuxKalmanMCMC = demoPosterior as PosteriorData;
