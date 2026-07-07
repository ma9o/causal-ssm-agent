import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage6Data } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { TreatmentRankingTable } from "./treatment-ranking-table";
import fixture from "../../__fixtures__/demo-run/stage-6.json";

const data = fixture as unknown as Stage6Data;

const meta = {
  title: "Pipeline/Stages/5b – Inference & Diagnostics/TreatmentRankingTable",
  component: TreatmentRankingTable,
  decorators: [withContainer()],
} satisfies Meta<typeof TreatmentRankingTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { results: data.intervention_results },
};
