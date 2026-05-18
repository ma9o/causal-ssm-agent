import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@nof1-causal-lab/api-types";
import type { Stage5bData } from "@nof1-causal-lab/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage5bContent from "./stage-5b-content";
import fixture from "../../../../../../data/DEMO_HEALTH/run/stage-5b.json";
import auxGibbsFixture from "../../../../../../data/DEMO_HEALTH/run/stage-5b-aux-gibbs.json";

const stage = STAGES.find((s) => s.id === "stage-5b")!;
const data = fixture as Stage5bData;
const auxGibbsData = auxGibbsFixture as Stage5bData;

const meta = {
  title: "Pipeline/Stages/5b – Inference & Diagnostics",
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
  outcome: data.outcome,
  elapsedMs: 124_500,
  renderContent: (args) => <Stage5bContent {...args} />,
});

export const CompletedAuxGibbs = createCompletedStageStory({
  name: "Completed (Aux Gibbs)",
  stage,
  args: { data: auxGibbsData, workspaceId: "demo-user" },
  outcome: auxGibbsData.outcome,
  elapsedMs: 342_000,
  renderContent: (args) => <Stage5bContent {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
