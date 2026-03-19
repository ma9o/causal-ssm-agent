import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage6Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { TreatmentRankingTable } from "./treatment-ranking-table";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-6.json";

const data = fixture as unknown as Stage6Data;

const meta = {
  title: "Stages/Inference/TreatmentRankingTable",
  component: TreatmentRankingTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof TreatmentRankingTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { results: data.intervention_results },
};
