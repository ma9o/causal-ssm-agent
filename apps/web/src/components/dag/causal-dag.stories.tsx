import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage1aData, Stage1bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { CausalDag } from "./causal-dag";
import stage1aFixture from "../../../../../data/DOCTOLIB/run/stage-1a.json";
import stage1bFixture from "../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage1a = stage1aFixture as Stage1aData;
const stage1b = stage1bFixture as Stage1bData;
const constructs = stage1a.latent_model.constructs;
const edges = stage1a.latent_model.edges;
const indicators = stage1b.causal_spec.measurement.indicators;
const identifiability = stage1b.causal_spec.identifiability;

const meta = {
  title: "DAG/CausalDag",
  component: CausalDag,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof CausalDag>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => <CausalDag constructs={constructs} edges={edges} />,
};

export const WithIndicators: Story = {
  render: () => (
    <CausalDag constructs={constructs} edges={edges} indicators={indicators} />
  ),
};

export const WithIdentifiability: Story = {
  render: () => (
    <CausalDag
      constructs={constructs}
      edges={edges}
      indicators={indicators}
      identifiability={identifiability}
    />
  ),
};

export const Empty: Story = {
  render: () => <CausalDag constructs={[]} edges={[]} />,
};
