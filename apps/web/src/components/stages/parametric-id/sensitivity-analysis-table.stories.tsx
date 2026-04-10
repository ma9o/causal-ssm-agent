import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage4bData } from "@causal-ssm/api-types";
import { withContainer } from "@/components/story-decorators";
import { SensitivityAnalysisTable } from "./sensitivity-analysis-table";
import fixture from "../../../../../../data/GOLDEN/run/stage-4b.json";

const data = fixture as unknown as Stage4bData;

const meta = {
  title: "Stages/ParametricId/SensitivityAnalysisTable",
  component: SensitivityAnalysisTable,
  decorators: [withContainer()],
} satisfies Meta<typeof SensitivityAnalysisTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { result: data.parametric_id.sensitivity_analysis! },
};
