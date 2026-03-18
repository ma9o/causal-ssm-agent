import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { PowerScalingTable } from "./power-scaling-table";
import nutsdaFixture from "../../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";

const data = nutsdaFixture as Stage5bData;

const meta = {
  title: "Stages/Inference/PowerScalingTable",
  component: PowerScalingTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof PowerScalingTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithPSIS: Story = {
  render: () => <PowerScalingTable results={data.power_scaling ?? []} />,
};

export const WithoutPSIS: Story = {
  render: () => (
    <PowerScalingTable
      results={(data.power_scaling ?? []).map(({ psis_k_hat, ...rest }) => rest) as Stage5bData["power_scaling"] & {}}
    />
  ),
};
