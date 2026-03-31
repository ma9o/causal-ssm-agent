import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { Stage1bData, Stage2Data, Stage4Data } from "@causal-ssm/api-types";
import {
  createCompletedStageStory,
  createOpenPanelStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage4Content from "./stage-4-content";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-4.json";
import stage2Fixture from "../../../../../../data/DOCTOLIB/run/stage-2.json";
import stage1bFixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage = STAGES.find((s) => s.id === "stage-4")!;
const data = fixture as unknown as Stage4Data;
const extractions = (stage2Fixture as unknown as Stage2Data).combined_extractions_sample;
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
  args: { data, extractions, indicators },
  outcome: data.outcome,
  elapsedMs: 15_600,
  trace: data.llm_trace ?? undefined,
  renderContent: (args) => <Stage4Content {...args} />,
});

export const OpenPanel = createOpenPanelStageStory({
  stage,
  args: { data, extractions, indicators },
  outcome: data.outcome,
  elapsedMs: 15_600,
  trace: data.llm_trace ?? undefined,
  renderContent: (args) => <Stage4Content {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
