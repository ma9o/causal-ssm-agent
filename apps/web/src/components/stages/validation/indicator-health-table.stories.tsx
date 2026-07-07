import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { normalizeStage3Data } from "./__fixtures__/normalize-stage3";
import { IndicatorHealthTable } from "./indicator-health-table";
import fixture from "../../__fixtures__/demo-run/stage-3.json";

const data = normalizeStage3Data(fixture);

const meta = {
  title: "Pipeline/Stages/3 – Validation/IndicatorHealthTable",
  component: IndicatorHealthTable,
  decorators: [withContainer()],
} satisfies Meta<typeof IndicatorHealthTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithIssues: Story = {
  args: { audits: data.indicators ?? {} },
};

export const AllClean: Story = {
  args: { audits: {} },
};
