import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import MeasurementStructureView from "./measurement-structure-view";
import fixture from "../../__fixtures__/demo-run/measurement_structure.json";

const output = TRANSITIONS.find((s) => s.id === "measurement_structure")!;
const data = fixture as unknown as MeasurementStructureViewData;

const meta = {
  title: "Pipeline/Outputs/Measurement Structure/Panel",
  component: MeasurementStructureView,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof MeasurementStructureView>;

export default meta;

export const Pending = createOutputStatusStory(output, "pending");

export const Running = createOutputStatusStory(output, "running");

export const Completed = createCompletedOutputStory({
  output,
  args: { data },
  elapsedMs: 18_900,
  trace: data.llm_trace ?? undefined,
  renderContent: (args) => <MeasurementStructureView {...args} />,
});

export const OpenPanel = createCompletedOutputStory({
  output,
  args: { data },
  elapsedMs: 18_900,
  trace: data.llm_trace ?? undefined,
  defaultPanelOpen: true,
  renderContent: (args) => <MeasurementStructureView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
