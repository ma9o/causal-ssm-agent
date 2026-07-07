import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@nof1-causal-lab/api-types";
import type { Stage5bData } from "@nof1-causal-lab/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage5bContent from "./stage-5b-content";
import fixture from "../../__fixtures__/demo-run/stage-5b.json";
import auxKalmanMCMCFixture from "../../__fixtures__/demo-run/stage-5b.json";

const stage = STAGES.find((s) => s.id === "stage-5b")!;
const data = fixture as Stage5bData;
const auxKalmanMCMCData = auxKalmanMCMCFixture as Stage5bData;

const meta = {
  title: "Pipeline/Stages/5b – Inference & Diagnostics/Panel",
  component: Stage5bContent,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage5bContent>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const CompletedMAP = createCompletedStageStory({
  name: "Completed (MAP)",
  stage,
  args: { data, workspaceId: "demo-user" },
  elapsedMs: 124_500,
  renderContent: (args) => <Stage5bContent {...args} />,
});

export const CompletedAuxKalmanMCMC = createCompletedStageStory({
  name: "Completed (Auxiliary Kalman MCMC)",
  stage,
  args: { data: auxKalmanMCMCData, workspaceId: "demo-user" },
  elapsedMs: 342_000,
  renderContent: (args) => <Stage5bContent {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
