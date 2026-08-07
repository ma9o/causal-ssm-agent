"use client";

import { GitBranch, Layers3 } from "lucide-react";
import { useState } from "react";
import { InteractiveDag } from "@/components/dag/interactive/interactive-dag";
import { SimulationViewer } from "@/components/dag/simulation-viewer";
import { StructureDag } from "@/components/dag/structure-dag";
import { Badge } from "@/components/ui/badge";
import {
  isModelNodeId,
  type ModelNodeId,
  PROTOTYPE_CONSTRUCTS,
  PROTOTYPE_EDGES,
  PROTOTYPE_INDICATORS,
  PROTOTYPE_MOCK_SCENARIOS,
  PROTOTYPE_NODE_STATUSES,
  PROTOTYPE_SCENARIOS,
  PROTOTYPE_SIMULATE,
  type WorkspaceLayerId,
} from "./artifact-workspace-fixture";

interface ModelCanvasProps {
  visibleLayers: ReadonlySet<WorkspaceLayerId>;
  selectedNode: ModelNodeId;
  onSelectNode: (node: ModelNodeId) => void;
}

export function ModelCanvas({ visibleLayers, selectedNode, onSelectNode }: ModelCanvasProps) {
  const measurementVisible = visibleLayers.has("model.measurement");
  const identificationVisible = visibleLayers.has("model.identification");
  const fittedVisible = visibleLayers.has("model.dynamics") || visibleLayers.has("model.posterior");
  const simulationVisible = visibleLayers.has("model.simulation");
  const [selectedScenario, setSelectedScenario] = useState<string | null>(
    PROTOTYPE_SCENARIOS.find((scenario) => scenario.provenance === "intervention")?.key ??
      PROTOTYPE_SCENARIOS[0]?.key ??
      null,
  );

  const selectConstruct = (constructName: string) => {
    if (isModelNodeId(constructName)) onSelectNode(constructName);
  };

  const statuses = identificationVisible ? PROTOTYPE_NODE_STATUSES : undefined;
  const indicators = measurementVisible ? PROTOTYPE_INDICATORS : [];

  return (
    <div className="min-h-0 bg-slate-50/60">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-white px-4 py-2.5">
        <div className="flex items-center gap-2">
          <GitBranch className="size-3.5 text-blue-600" />
          <span className="text-xs font-medium text-slate-700">Unified causal model</span>
          <span className="hidden text-[10px] text-muted-foreground sm:inline">
            selected: {selectedNode.replaceAll("_", " ")}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="gap-1 font-mono text-[9px]">
            <Layers3 className="size-3" />
            {simulationVisible
              ? "SimulationViewer"
              : fittedVisible
                ? "InteractiveDag"
                : "StructureDag"}
          </Badge>
          {identificationVisible ? <Badge variant="warning">1 marginalized</Badge> : null}
        </div>
      </div>

      <div className="p-3">
        {simulationVisible ? (
          <SimulationViewer
            scenarios={PROTOTYPE_SCENARIOS}
            graph={{
              constructs: PROTOTYPE_CONSTRUCTS,
              edges: PROTOTYPE_EDGES,
              indicators,
              indicatorsVisible: measurementVisible,
              nodeStatuses: statuses,
            }}
            selectedKey={selectedScenario}
            onSelect={setSelectedScenario}
            onSimulate={PROTOTYPE_SIMULATE}
            onNodeClick={selectConstruct}
          />
        ) : fittedVisible ? (
          <InteractiveDag
            constructs={PROTOTYPE_CONSTRUCTS}
            edges={PROTOTYPE_EDGES}
            indicators={indicators}
            indicatorsVisible={measurementVisible}
            nodeStatuses={statuses}
            result={PROTOTYPE_MOCK_SCENARIOS.baseline.result}
            onNodeClick={selectConstruct}
          />
        ) : (
          <StructureDag
            constructs={PROTOTYPE_CONSTRUCTS}
            edges={PROTOTYPE_EDGES}
            indicators={indicators}
            nodeStatuses={statuses}
            onNodeClick={selectConstruct}
          />
        )}
      </div>
    </div>
  );
}
