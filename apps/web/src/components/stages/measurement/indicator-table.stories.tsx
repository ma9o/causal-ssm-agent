import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage1bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { IndicatorTable } from "./indicator-table";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";

const data = fixture as unknown as Stage1bData;
const indicators = data.causal_spec.measurement.indicators;

const meta = {
  title: "Stages/Measurement/IndicatorTable",
  component: IndicatorTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof IndicatorTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { indicators },
};

export const Empty: Story = {
  args: { indicators: [] },
};
