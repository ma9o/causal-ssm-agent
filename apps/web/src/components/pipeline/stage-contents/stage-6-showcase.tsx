"use client";

import { CausalDag } from "@/components/dag/causal-dag";
import { InterventionDag } from "@/components/dag/intervention-dag";
import type {
  EdgePosterior,
  Stage6SimulationResult,
} from "@/components/dag/intervention-dag-types";
import type { CausalEdge, Construct, Indicator, Stage6Data } from "@nof1-causal-lab/api-types";
import Stage6Content from "./stage-6-content";

type Stage6DagSceneBase = {
  height?: string;
};

export type Stage6BaselineDagScene = Stage6DagSceneBase & {
  kind: "baseline";
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
};

export type Stage6SimulationDagScene = Stage6DagSceneBase & {
  kind: "simulation";
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  edgePosteriors?: Record<string, EdgePosterior>;
  requestedHorizonDays?: number;
  simulationResult: Stage6SimulationResult;
};

export type Stage6DagScene = Stage6BaselineDagScene | Stage6SimulationDagScene;

export function createStage6BaselineDagScene(
  scene: Omit<Stage6BaselineDagScene, "kind">,
): Stage6BaselineDagScene {
  return { kind: "baseline", ...scene };
}

export function createStage6SimulationDagScene(
  scene: Omit<Stage6SimulationDagScene, "kind">,
): Stage6SimulationDagScene {
  return { kind: "simulation", ...scene };
}

export default function Stage6Showcase({
  data,
  dagScene,
}: {
  data: Stage6Data;
  dagScene?: Stage6DagScene;
}) {
  return (
    <div className="space-y-4">
      <Stage6Content data={data} />
      {dagScene ? (
        dagScene.kind === "baseline" ? (
          <CausalDag
            constructs={dagScene.constructs}
            edges={dagScene.edges}
            indicators={dagScene.indicators}
            height={dagScene.height ?? "600px"}
          />
        ) : (
          <InterventionDag
            constructs={dagScene.constructs}
            edges={dagScene.edges}
            indicators={dagScene.indicators}
            edgePosteriors={dagScene.edgePosteriors}
            requestedHorizonDays={dagScene.requestedHorizonDays}
            simulationResult={dagScene.simulationResult}
            height={dagScene.height ?? "600px"}
          />
        )
      ) : null}
    </div>
  );
}
