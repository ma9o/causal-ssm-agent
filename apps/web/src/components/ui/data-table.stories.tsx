import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage2Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DataTable } from "./data-table";
import fixture from "../../../../../data/DOCTOLIB/run/stage-2.json";

const data = fixture as Stage2Data;
const rows = data.combined_extractions_sample.slice(0, 50);

const meta = {
  title: "UI/DataTable",
  component: DataTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof DataTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => <DataTable rows={rows} />,
};

export const WithTooltips: Story = {
  render: () => (
    <DataTable
      rows={rows}
      columnTooltips={{
        indicator: "The measurement indicator name",
        value: "The extracted value",
        timestamp: "When this observation was recorded",
      }}
    />
  ),
};

export const CustomHeight: Story = {
  render: () => <DataTable rows={rows} maxHeight="max-h-96" />,
};

export const Empty: Story = {
  render: () => <DataTable rows={[]} />,
};
