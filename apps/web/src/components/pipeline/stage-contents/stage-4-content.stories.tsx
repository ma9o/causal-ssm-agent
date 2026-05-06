import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage1bData } from "@causal-ssm/api-types";
import {
  createCompletedStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage4Content from "./stage-4-content";
import { stage4Data } from "@/components/stages/model-spec/__fixtures__/model-spec-fixtures";
import stage1bFixture from "../../../../../../data/DEMO_HEALTH/run/stage-1b.json";

const stage = STAGES.find((s) => s.id === "stage-4")!;
const indicators = (stage1bFixture as unknown as Stage1bData).causal_spec.measurement.indicators;

const meta = {
  title: "Pipeline/Stages/4 – Model Specification",
  component: Stage4Content,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage4Content>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export { StateMachineReplay as Running } from "./stage-4-running-content.stories";

export const Completed = createCompletedStageStory({
  stage,
  args: { data: stage4Data, indicators },
  outcome: stage4Data.outcome,
  elapsedMs: 15_600,
  trace: stage4Data.llm_trace ?? undefined,
  renderContent: (args) => <Stage4Content {...args} />,
});

export const OpenPanel = createCompletedStageStory({
  stage,
  args: { data: stage4Data, indicators },
  outcome: stage4Data.outcome,
  elapsedMs: 15_600,
  trace: stage4Data.llm_trace ?? undefined,
  defaultPanelOpen: true,
  renderContent: (args) => <Stage4Content {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
