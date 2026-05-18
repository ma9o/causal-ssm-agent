import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@nof1-causal-lab/api-types";
import type { Stage5aData } from "@nof1-causal-lab/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage5aContent from "./stage-5a-content";
import fixture from "../../../../../../data/DEMO_HEALTH/run/stage-5a.json";

const stage = STAGES.find((s) => s.id === "stage-5a")!;
const data = fixture as Stage5aData;

const meta = {
  title: "Pipeline/Stages/5a – SVI Preflight",
  component: Stage5aContent,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage5aContent>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Completed = createCompletedStageStory({
  stage,
  args: { data },
  outcome: data.outcome,
  elapsedMs: 32_100,
  renderContent: (args) => <Stage5aContent {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
