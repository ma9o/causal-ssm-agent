import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage0Data } from "@causal-ssm/api-types";
import { withContainer } from "@/components/story-decorators";
import { DataSummaryStats } from "./data-summary-stats";
import fixture from "../../../../../../data/DEMO_HEALTH/run/stage-0.json";

const data = fixture as Stage0Data;

const meta = {
  title: "Stages/Preprocess/DataSummaryStats",
  component: DataSummaryStats,
  decorators: [withContainer("max-w-3xl")],
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
