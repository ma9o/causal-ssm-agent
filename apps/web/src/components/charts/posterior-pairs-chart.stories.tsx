import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { PosteriorPairsChart } from "./posterior-pairs-chart";
import fixture from "../../../../../data/DOCTOLIB/run/stage-5b.json";

const data = fixture as Stage5bData;
const pairs = data.posterior_pairs ?? [];

const meta = {
  title: "Charts/PosteriorPairsChart",
  component: PosteriorPairsChart,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-sm mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof PosteriorPairsChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => <PosteriorPairsChart pair={pairs[0]} />,
};
