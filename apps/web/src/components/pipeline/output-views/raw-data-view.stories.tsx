import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { RawDataData } from "@nof1-causal-lab/api-types";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import RawDataView from "./raw-data-view";
import fixture from "../../__fixtures__/demo-run/raw_data.json";

const output = TRANSITIONS.find((s) => s.id === "raw_data")!;
const data = fixture as unknown as RawDataData;
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
  trace: data.llm_trace ?? undefined,
  renderContent: (args) => <RawDataView {...args} />,
});

export const OpenPanel = createCompletedOutputStory({
  output,
  args: { data, workspaceId },
  elapsedMs: 4_320,
  trace: data.llm_trace ?? undefined,
  defaultPanelOpen: true,
  renderContent: (args) => <RawDataView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
