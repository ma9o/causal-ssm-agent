import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { RawDataData } from "@nof1-causal-lab/api-types";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import RawDataView from "./raw-data-view";
import { demoRawData } from "../../__fixtures__/demo-artifacts";
import { demoTraces } from "../../__fixtures__/demo-traces";

const output = TRANSITIONS.find((s) => s.id === "raw_data")!;
const data = demoRawData as RawDataData;
const workspaceId = "demo-user";

const meta = {
  title: "Pipeline/Outputs/Raw Data/Panel",
  component: RawDataView,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof RawDataView>;

export default meta;

export const Pending = createOutputStatusStory(output, "pending");

export const Running = createOutputStatusStory(output, "running");

export const Completed = createCompletedOutputStory({
  output,
  args: { data, workspaceId },
  elapsedMs: 4_320,
  trace: demoTraces.raw_data,
  renderContent: (args) => <RawDataView {...args} />,
});

export const OpenPanel = createCompletedOutputStory({
  output,
  args: { data, workspaceId },
  elapsedMs: 4_320,
  defaultPanelOpen: true,
  trace: demoTraces.raw_data,
  renderContent: (args) => <RawDataView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
