import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import { normalizeStage3Data } from "@/components/stages/validation/__fixtures__/normalize-stage3";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage3Content, { Stage3FixAction } from "./stage-3-content";
import fixture from "../../../../../../data/DEMO_HEALTH/run/stage-3.json";

const stage = STAGES.find((s) => s.id === "stage-3")!;

const data = normalizeStage3Data(fixture);

const meta = {
  title: "Pipeline/Stages/3 – Validation",
  component: Stage3Content,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage3Content>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Completed = createCompletedStageStory({
  stage,
  args: { data },
  outcome: data.outcome,
  elapsedMs: 3_800,
  renderShellProps: (args) => ({
    actions: <Stage3FixAction data={args.data} onFix={() => undefined} />,
  }),
  renderContent: (args) => <Stage3Content {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
