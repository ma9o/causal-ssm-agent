import type { CausalEdge, Construct, Indicator } from "@nof1-causal-lab/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { useState } from "react";
import { type ConstructStatus, StructureDag } from "./structure-dag";

function construct(
  name: string,
  role: Construct["role"],
  opts: { outcome?: boolean; invariant?: boolean } = {},
): Construct {
  return {
    name,
    description: `${name} construct`,
    role,
    is_outcome: opts.outcome ?? false,
    temporal_status: opts.invariant ? "time_invariant" : "time_varying",
  };
}

function edge(cause: string, effect: string, lagged = false): CausalEdge {
  return { cause, effect, description: `${cause} → ${effect}`, lagged, sources: [] };
}

function indicator(name: string, constructName: string, dtype: string): Indicator {
  return {
    name,
    construct_name: constructName,
    how_to_measure: "",
    construct_polarity: "positive",
    measurement_dtype: dtype,
    aggregation: "mean",
    source_columns: [],
    extraction_mode: "computed",
    support_kind: "interval",
    summary_operator: "mean",
    anchor_policy: "support_end",
  };
}

const CONSTRUCTS: Construct[] = [
  construct("life_events_load", "exogenous", { invariant: true }),
  construct("adherence", "endogenous"),
  construct("serotonergic_exposure", "endogenous"),
  construct("physical_activity", "endogenous"),
  construct("sleep_quality", "endogenous"),
  construct("affective_state", "endogenous", { outcome: true }),
];

const EDGES: CausalEdge[] = [
  edge("adherence", "serotonergic_exposure"),
  edge("life_events_load", "affective_state"),
  edge("serotonergic_exposure", "sleep_quality"),
  edge("serotonergic_exposure", "affective_state"),
  edge("physical_activity", "sleep_quality"),
  edge("sleep_quality", "affective_state"),
  edge("affective_state", "sleep_quality", true), // lagged feedback
];

const INDICATORS: Indicator[] = [
  indicator("ssri_dose_mg", "serotonergic_exposure", "continuous"),
  indicator("self_reported_mood", "affective_state", "ordinal"),
  indicator("phq9", "affective_state", "count"),
  indicator("sleep_hours", "sleep_quality", "continuous"),
  indicator("step_count", "physical_activity", "count"),
  indicator("med_taken", "adherence", "binary"),
];

const STATUSES: Record<string, ConstructStatus> = {
  life_events_load: "blocking",
  serotonergic_exposure: "blocking",
  physical_activity: "marginalized",
};

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
        <StructureDag constructs={CONSTRUCTS} edges={EDGES} onNodeClick={setSelected} />
        <p className="mt-2 text-sm text-muted-foreground">selected: {selected ?? "—"}</p>
      </div>
    );
  },
};

/** measurement-structure — measurement + identifiability: indicators, status borders, blocking edges. */
export const MeasurementStructureStory: Story = {
  render: () => (
    <div className="p-4">
      <StructureDag
        constructs={CONSTRUCTS}
        edges={EDGES}
        indicators={INDICATORS}
        nodeStatuses={STATUSES}
      />
    </div>
  ),
};
