import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage4bData } from "@causal-ssm/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage4bContent from "./stage-4b-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-4b.json";

const stage = STAGES.find((s) => s.id === "stage-4b")!;
const data = fixture as Stage4bData;

const meta = {
  title: "Pipeline/Stages/4b – Parametric ID",
  component: Stage4bContent,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage4bContent>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Completed = createCompletedStageStory({
  stage,
  args: { data },
  outcome: data.outcome,
  elapsedMs: 8_400,
  renderContent: (args) => <Stage4bContent {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
