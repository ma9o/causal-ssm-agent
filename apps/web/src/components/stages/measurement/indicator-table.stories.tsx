import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage1bData } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { IndicatorTable } from "./indicator-table";
import fixture from "../../../../../../data/DEMO/run/stage-1b.json";

const data = fixture as unknown as Stage1bData;
const indicators = data.causal_spec.measurement.indicators;

const meta = {
  title: "Pipeline/Stages/1b – Measurement/IndicatorTable",
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
