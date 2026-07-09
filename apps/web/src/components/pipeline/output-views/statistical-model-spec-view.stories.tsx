import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { MeasurementStructureViewData } from "@nof1-causal-lab/api-types";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import StatisticalModelSpecView from "./statistical-model-spec-view";
import { demoTraces } from "@/components/__fixtures__/demo-traces";
import { demoMeasurementStructure } from "@/components/__fixtures__/demo-artifacts";
import { modelSpecData } from "@/components/analysis-widgets/statistical-model-spec/__fixtures__/statistical-model-spec-fixtures";

const output = TRANSITIONS.find((s) => s.id === "statistical_model_spec")!;
const indicators = (demoMeasurementStructure as MeasurementStructureViewData).causal_design
  .measurement.indicators;

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/Panel",
  component: StatisticalModelSpecView,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof StatisticalModelSpecView>;

export default meta;

export const Pending = createOutputStatusStory(output, "pending");

export { AdmissionReplay as Running } from "./statistical-model-spec-running-view.stories";

export const Completed = createCompletedOutputStory({
  output,
  args: { data: modelSpecData, indicators },
  elapsedMs: 15_600,
  trace: demoTraces.statistical_model_spec,
  renderContent: (args) => <StatisticalModelSpecView {...args} />,
});

export const OpenPanel = createCompletedOutputStory({
  output,
  args: { data: modelSpecData, indicators },
  elapsedMs: 15_600,
  defaultPanelOpen: true,
  trace: demoTraces.statistical_model_spec,
  renderContent: (args) => <StatisticalModelSpecView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
