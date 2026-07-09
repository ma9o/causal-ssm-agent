import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import MeasurementStructureView from "./measurement-structure-view";
import { demoMeasurementStructure } from "../../__fixtures__/demo-artifacts";
import { demoTraces } from "../../__fixtures__/demo-traces";

const output = TRANSITIONS.find((s) => s.id === "measurement_structure")!;
const data = demoMeasurementStructure as MeasurementStructureViewData;

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
  trace: demoTraces.measurement_structure,
  renderContent: (args) => <MeasurementStructureView {...args} />,
});

export const OpenPanel = createCompletedOutputStory({
  output,
  args: { data },
  elapsedMs: 18_900,
  defaultPanelOpen: true,
  trace: demoTraces.measurement_structure,
  renderContent: (args) => <MeasurementStructureView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
