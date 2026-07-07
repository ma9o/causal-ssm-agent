import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@nof1-causal-lab/api-types";
import type { Stage0Data } from "@nof1-causal-lab/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage0Content from "./stage-0-content";
import fixture from "../../__fixtures__/demo-run/stage-0.json";

const stage = STAGES.find((s) => s.id === "stage-0")!;
const data = fixture as unknown as Stage0Data;
const workspaceId = "demo-user";

const meta = {
  title: "Pipeline/Stages/0 – Preprocess/Panel",
  component: Stage0Content,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage0Content>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Completed = createCompletedStageStory({
  stage,
  args: { data, workspaceId },
  elapsedMs: 4_320,
  trace: data.llm_trace ?? undefined,
  renderContent: (args) => <Stage0Content {...args} />,
});

export const OpenPanel = createCompletedStageStory({
  stage,
  args: { data, workspaceId },
  elapsedMs: 4_320,
  trace: data.llm_trace ?? undefined,
  defaultPanelOpen: true,
  renderContent: (args) => <Stage0Content {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
