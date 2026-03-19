import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { PosteriorDensityChart } from "./posterior-density-chart";
import fixture from "../../../../../data/DOCTOLIB/run/stage-5b.json";

const data = fixture as Stage5bData;
const marginals = data.posterior_marginals ?? [];

const meta = {
  title: "Charts/PosteriorDensityChart",
  component: PosteriorDensityChart,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-sm mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof PosteriorDensityChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { marginal: marginals[0] },
};
