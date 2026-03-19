import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage0Data } from "@causal-ssm/api-types";
import { DataSummaryStats } from "./data-summary-stats";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-0.json";

const data = fixture as Stage0Data;

const meta = {
  title: "Stages/Preprocess/DataSummaryStats",
  component: DataSummaryStats,
  decorators: [
    (Story) => (
      <div className="max-w-3xl mx-auto p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof DataSummaryStats>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    nRecords: data.n_records,
    nColumns: data.n_columns,
    dateRange: data.date_range,
  },
};

export const WithoutColumns: Story = {
  args: {
    nRecords: data.n_records,
    dateRange: data.date_range,
  },
};
