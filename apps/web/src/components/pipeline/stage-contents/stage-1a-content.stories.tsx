import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@nof1-causal-lab/api-types";
import type { Stage1aData } from "@nof1-causal-lab/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage1aContent from "./stage-1a-content";
import fixture from "../../__fixtures__/demo-run/stage-1a.json";

const stage = STAGES.find((s) => s.id === "stage-1a")!;
const data = fixture as unknown as Stage1aData;

const meta = {
  title: "Pipeline/Stages/1a – Latent Structure/Panel",
  component: Stage1aContent,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage1aContent>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Completed = createCompletedStageStory({
  stage,
  args: { data },
  elapsedMs: 12_450,
  trace: data.llm_trace ?? undefined,
  renderContent: (args) => <Stage1aContent {...args} />,
});

export const OpenPanel = createCompletedStageStory({
  stage,
  args: { data },
  elapsedMs: 12_450,
  trace: data.llm_trace ?? undefined,
  defaultPanelOpen: true,
  renderContent: (args) => <Stage1aContent {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
