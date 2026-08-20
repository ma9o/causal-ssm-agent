import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import {
  constructs,
  design,
  edges,
  indicators,
  knownInputs,
  structuralPlan,
} from "./__fixtures__/dag-base-fixtures";
import { deriveConstructStatuses } from "./construct-statuses";
import { StructureDag } from "./structure-dag";

const nodeStatuses = deriveConstructStatuses(design, structuralPlan);

const meta: Meta<typeof StructureDag> = {
  title: "DAG/Structure DAG",
  component: StructureDag,
  parameters: { layout: "fullscreen" },
};
export default meta;

type Story = StoryObj<typeof StructureDag>;

/** latent-structure — latent structure: clickable constructs, no indicators or statuses. */
export const LatentStructureStory: Story = {
  render: () => {
    const [selected, setSelected] = useState<string | null>(null);
    return (
      <div className="p-4">
        <StructureDag constructs={constructs} edges={edges} onNodeClick={setSelected} />
        <p className="mt-2 text-sm text-muted-foreground">selected: {selected ?? "—"}</p>
      </div>
    );
  },
};

/** Canonical causal-design fixture with measurement and backend dispositions overlaid. */
export const MeasurementStructureStory: Story = {
  render: () => (
    <div className="p-4">
      <StructureDag
        constructs={constructs}
        edges={edges}
        indicators={indicators}
        knownInputs={knownInputs}
        nodeStatuses={nodeStatuses}
      />
    </div>
  ),
};
