"use client";

import { StructureDag } from "@/components/dag/structure-dag";
import { ConstructDetailPanel } from "@/components/analysis-widgets/latent-structure/construct-detail-panel";
import { EdgeList } from "@/components/analysis-widgets/latent-structure/edge-list";
import type { LatentStructureData } from "@nof1-causal-lab/api-types";
import { useState } from "react";

export default function LatentStructureView({ data }: { data: LatentStructureData }) {
  const [selectedConstruct, setSelectedConstruct] = useState<string | null>(null);
  const selected = data.latent_structure.constructs.find((c) => c.name === selectedConstruct);

  return (
    <div className="space-y-4">
      <StructureDag
        constructs={data.latent_structure.constructs}
        edges={data.latent_structure.edges}
        onNodeClick={setSelectedConstruct}
      />
      {selected && <ConstructDetailPanel construct={selected} />}
      <EdgeList edges={data.latent_structure.edges} />
    </div>
  );
}
