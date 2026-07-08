import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { BaselineReportData } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { TreatmentRankingTable } from "./treatment-ranking-table";
import fixture from "../../__fixtures__/demo-run/baseline_report.json";

const data = fixture as unknown as BaselineReportData;

const meta = {
  title: "Pipeline/Outputs/Posterior/TreatmentRankingTable",
  component: TreatmentRankingTable,
  decorators: [withContainer()],
} satisfies Meta<typeof TreatmentRankingTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { results: data.intervention_results },
};
