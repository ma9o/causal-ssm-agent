import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { normalizeValidationReportData } from "./__fixtures__/normalize-validation-report";
import { IndicatorHealthTable } from "./indicator-health-table";
import { demoValidationReport } from "../../__fixtures__/demo-artifacts";

const data = normalizeValidationReportData(demoValidationReport);

const meta = {
  title: "Pipeline/Outputs/Validation Report/IndicatorHealthTable",
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
