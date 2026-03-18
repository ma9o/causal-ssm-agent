import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { LOOPITChart } from "./loo-pit-chart";
import fixture from "../../../../../data/DOCTOLIB/run/stage-5b.json";

const data = fixture as Stage5bData;

const meta = {
  title: "Charts/LOOPITChart",
  component: LOOPITChart,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-md mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof LOOPITChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => <LOOPITChart loo={data.loo_diagnostics!} />,
};
