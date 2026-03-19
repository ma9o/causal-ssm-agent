import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5aData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ELBOLossChart } from "./elbo-loss-chart";
import fixture from "../../../../../data/DOCTOLIB/run/stage-5a.json";

const data = fixture as Stage5aData;

const meta = {
  title: "Charts/ELBOLossChart",
  component: ELBOLossChart,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof ELBOLossChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { diagnostics: data.svi_diagnostics! },
};
