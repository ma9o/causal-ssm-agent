import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage5bData } from "@causal-ssm/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage5bContent from "./stage-5b-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-5b.json";
import nutsdaFixture from "../../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";

const stage = STAGES.find((s) => s.id === "stage-5b")!;
const data = fixture as Stage5bData;
const nutsdaData = nutsdaFixture as Stage5bData;

const meta = {
  title: "Pipeline/Stages/5b – Inference & Diagnostics",
  component: Stage5bContent,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage5bContent>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const CompletedSVI = createCompletedStageStory({
  name: "Completed (SVI / Laplace EM)",
  stage,
  args: { data, workspaceId: "demo-user" },
  outcome: data.outcome,
  elapsedMs: 124_500,
  renderContent: (args) => <Stage5bContent {...args} />,
});

export const CompletedNUTS = createCompletedStageStory({
  name: "Completed (NUTS / DA)",
  stage,
  args: { data: nutsdaData, workspaceId: "demo-user" },
  outcome: nutsdaData.outcome,
  elapsedMs: 342_000,
  renderContent: (args) => <Stage5bContent {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
