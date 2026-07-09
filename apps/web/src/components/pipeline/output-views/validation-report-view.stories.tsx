import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import { normalizeValidationReportData } from "@/components/analysis-widgets/validation-report/__fixtures__/normalize-validation-report";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import ValidationReportView from "./validation-report-view";
import { demoValidationReport } from "../../__fixtures__/demo-artifacts";

const output = TRANSITIONS.find((s) => s.id === "validation_report")!;

const data = normalizeValidationReportData(demoValidationReport);

const meta = {
  title: "Pipeline/Outputs/Validation Report/Panel",
  component: ValidationReportView,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof ValidationReportView>;

export default meta;

export const Pending = createOutputStatusStory(output, "pending");

export const Running = createOutputStatusStory(output, "running");

export const Completed = createCompletedOutputStory({
  output,
  args: { data },
  elapsedMs: 3_800,
  renderContent: (args) => <ValidationReportView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
