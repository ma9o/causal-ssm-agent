import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage4bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SensitivityAnalysisTable } from "./sensitivity-analysis-table";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-4b.json";

const data = fixture as Stage4bData;

const meta = {
  title: "Stages/ParametricId/SensitivityAnalysisTable",
  component: SensitivityAnalysisTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof SensitivityAnalysisTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => (
    <SensitivityAnalysisTable result={data.parametric_id.sensitivity_analysis} />
  ),
};
