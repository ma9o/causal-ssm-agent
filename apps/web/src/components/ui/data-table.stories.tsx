import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage2Data } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { DataTable } from "./data-table";
import fixture from "../../../../../data/DEMO/run/stage-2.json";

const data = fixture as unknown as Stage2Data;
const rows = data.combined_extractions_sample.slice(0, 50);

const meta = {
  title: "UI/DataTable",
  component: DataTable,
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof DataTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { rows },
};

export const WithTooltips: Story = {
  args: {
    rows,
    columnTooltips: {
      indicator: "The measurement indicator name",
      value: "The extracted value",
      anchor_time: "When this observation was attached to the latent timeline",
    },
  },
};

export const CustomHeight: Story = {
  args: { rows, maxHeight: "max-h-96" },
};

export const Empty: Story = {
  args: { rows: [] },
};
