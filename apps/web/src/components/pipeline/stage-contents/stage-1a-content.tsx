"use client";

import { StructureDag } from "@/components/dag/structure-dag";
import { ConstructDetailPanel } from "@/components/stages/latent-model/construct-detail-panel";
import { EdgeList } from "@/components/stages/latent-model/edge-list";
import type { Stage1aData } from "@nof1-causal-lab/api-types";
import { useState } from "react";

export default function Stage1aContent({ data }: { data: Stage1aData }) {
  const [selectedConstruct, setSelectedConstruct] = useState<string | null>(null);
  const selected = data.latent_model.constructs.find((c) => c.name === selectedConstruct);

  return (
    <div className="space-y-4">
      <StructureDag
        constructs={data.latent_model.constructs}
        edges={data.latent_model.edges}
        onNodeClick={setSelectedConstruct}
      />
      {selected && <ConstructDetailPanel construct={selected} />}
      <EdgeList edges={data.latent_model.edges} />
    </div>
  );
}
