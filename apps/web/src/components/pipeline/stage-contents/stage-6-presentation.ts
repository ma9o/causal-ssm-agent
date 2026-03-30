import type { Stage1aData, Stage1bData } from "@causal-ssm/api-types";
import type { UIMessage } from "ai";
import { extractLatestStage6SimulationResult } from "./stage-6-follow-up";
import {
  createStage6BaselineDagScene,
  createStage6SimulationDagScene,
  type Stage6DagScene,
} from "./stage-6-showcase";

export function buildStage6DagScene({
  stage1a,
  stage1b,
  refinementMessages = [],
  height = "600px",
}: {
  stage1a?: Stage1aData;
  stage1b?: Stage1bData;
  refinementMessages?: UIMessage[];
  height?: string;
}): Stage6DagScene | undefined {
  if (!stage1a) {
    return undefined;
  }

  const sceneBase = {
    constructs: stage1a.latent_model.constructs,
    edges: stage1a.latent_model.edges,
    indicators: stage1b?.causal_spec.measurement.indicators,
    height,
  };
  const simulationResult = extractLatestStage6SimulationResult(refinementMessages);

  return simulationResult
    ? createStage6SimulationDagScene({
        ...sceneBase,
        simulationResult,
      })
    : createStage6BaselineDagScene(sceneBase);
}
