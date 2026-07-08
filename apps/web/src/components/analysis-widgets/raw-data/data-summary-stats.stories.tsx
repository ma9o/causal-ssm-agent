import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { RawDataData } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { DataSummaryStats } from "./data-summary-stats";
import fixture from "../../__fixtures__/demo-run/raw_data.json";

const data = fixture as unknown as RawDataData;

const meta = {
  title: "Pipeline/Outputs/Raw Data/DataSummaryStats",
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
