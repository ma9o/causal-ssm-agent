import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@nof1-causal-lab/api-types";
import type { Stage1bData } from "@nof1-causal-lab/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage1bContent from "./stage-1b-content";
import fixture from "../../../../../../data/DEMO/run/stage-1b.json";

const stage = STAGES.find((s) => s.id === "stage-1b")!;
const data = fixture as unknown as Stage1bData;

const meta = {
  title: "Pipeline/Stages/1b – Measurement/Panel",
  component: Stage1bContent,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage1bContent>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Completed = createCompletedStageStory({
  stage,
  args: { data },
  elapsedMs: 18_900,
  trace: data.llm_trace ?? undefined,
  renderContent: (args) => <Stage1bContent {...args} />,
});

export const OpenPanel = createCompletedStageStory({
  stage,
  args: { data },
  elapsedMs: 18_900,
  trace: data.llm_trace ?? undefined,
  defaultPanelOpen: true,
  renderContent: (args) => <Stage1bContent {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
