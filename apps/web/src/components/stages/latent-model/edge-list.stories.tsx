import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage1aData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { EdgeList } from "./edge-list";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-1a.json";

const data = fixture as Stage1aData;
const edges = data.latent_model.edges;

const meta = {
  title: "Stages/LatentModel/EdgeList",
  component: EdgeList,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-md mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof EdgeList>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => <EdgeList edges={edges} />,
};

export const Empty: Story = {
  render: () => <EdgeList edges={[]} />,
};
