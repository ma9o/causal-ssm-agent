import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { IndicatorTable } from "./indicator-table";
import { demoMeasurementStructure } from "../../__fixtures__/demo-artifacts";

const data = demoMeasurementStructure as MeasurementStructureViewData;
const indicators = data.causal_design.measurement.indicators;

const meta = {
  title: "Pipeline/Outputs/Measurement Structure/IndicatorTable",
  component: IndicatorTable,
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof IndicatorTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { indicators },
};

export const Empty: Story = {
  args: { indicators: [] },
};
