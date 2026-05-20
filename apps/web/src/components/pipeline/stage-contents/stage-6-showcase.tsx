"use client";

import { InterventionDag } from "@/components/dag/intervention-dag";
import type {
  EdgePosterior,
  Stage6SimulationResult,
} from "@/components/dag/intervention-dag-types";
import type { CausalEdge, Construct, Indicator, Stage6Data } from "@nof1-causal-lab/api-types";
import Stage6Content from "./stage-6-content";

export interface Stage6DagScene {
  constructs: Construct[];
  edges: CausalEdge[];
  indicators?: Indicator[];
  edgePosteriors?: Record<string, EdgePosterior>;
  requestedHorizonDays?: number;
  simulationResult?: Stage6SimulationResult;
  height?: string;
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
        <InterventionDag
          constructs={dagScene.constructs}
          edges={dagScene.edges}
          indicators={dagScene.indicators}
          edgePosteriors={dagScene.edgePosteriors}
          requestedHorizonDays={dagScene.requestedHorizonDays}
          simulationResult={dagScene.simulationResult}
          height={dagScene.height ?? "600px"}
        />
      ) : null}
    </div>
  );
}
