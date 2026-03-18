import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { PowerScalingScatter } from "./power-scaling-scatter";
import fixture from "../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";

const data = fixture as Stage5bData;

const meta = {
  title: "Charts/PowerScalingScatter",
  component: PowerScalingScatter,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-md mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof PowerScalingScatter>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => <PowerScalingScatter results={data.power_scaling ?? []} />,
};
