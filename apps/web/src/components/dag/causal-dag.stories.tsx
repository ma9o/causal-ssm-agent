import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage1aData, Stage1bData } from "@causal-ssm/api-types";
import { CausalDag } from "./causal-dag";
import type { ConstructStatus } from "./construct-node";
import stage1aFixture from "../../../../../data/DOCTOLIB/run/stage-1a.json";
import stage1bFixture from "../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage1a = stage1aFixture as unknown as Stage1aData;
const stage1b = stage1bFixture as unknown as Stage1bData;
const constructs = stage1a.latent_model.constructs;
const edges = stage1a.latent_model.edges;
const spec = stage1b.causal_spec;
const indicators = spec.measurement.indicators;

/** Derive node statuses from identifiability data (mirrors useNodeStatuses logic). */
function deriveNodeStatuses(): Record<string, ConstructStatus> {
  const statuses: Record<string, ConstructStatus> = {};

  const marginalized = new Set<string>();
  for (const s of Object.values(spec.identifiability?.identifiable_treatments ?? {})) {
    for (const c of s?.marginalized_confounders ?? []) marginalized.add(c);
  }

  const blocking = new Set<string>();
  for (const s of Object.values(spec.identifiability?.non_identifiable_treatments ?? {})) {
    for (const c of s?.confounders ?? []) blocking.add(c);
  }

  for (const c of spec.latent.constructs) {
    if (blocking.has(c.name)) statuses[c.name] = "blocking";
    else if (marginalized.has(c.name)) statuses[c.name] = "marginalized";
    else statuses[c.name] = "observed";
  }
  return statuses;
}

const nodeStatuses = deriveNodeStatuses();

const meta = {
  title: "DAG/CausalDag",
  component: CausalDag,
  decorators: [
    (Story) => (
      <div className="max-w-3xl mx-auto p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof CausalDag>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { constructs, edges },
};

export const WithIndicators: Story = {
  args: { constructs, edges, indicators },
};

export const WithNodeStatuses: Story = {
  args: { constructs, edges, indicators, nodeStatuses },
};

export const Empty: Story = {
  args: { constructs: [], edges: [] },
};
