import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { EnergyChart } from "./energy-chart";
import fixture from "../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";

const data = fixture as Stage5bData;

const meta = {
  title: "Charts/EnergyChart",
  component: EnergyChart,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof EnergyChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { energy: data.mcmc_diagnostics!.energy! },
};
