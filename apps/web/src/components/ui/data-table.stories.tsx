import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { measurementsData } from "@/components/__fixtures__/measurements-data";
import { DataTable } from "./data-table";

const rows = measurementsData.combined_extractions_sample.slice(0, 50);

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
