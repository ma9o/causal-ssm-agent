import type { Meta } from "@storybook/nextjs-vite";
import { STAGES } from "@causal-ssm/api-types";
import type { LLMTrace, Stage1aData, Stage1bData, Stage6Data } from "@causal-ssm/api-types";
import { CausalDag } from "@/components/dag/causal-dag";
import {
  createCompletedStageStory,
  createOpenPanelStageStory,
  createStageStatusStory,
  stageStoryDecorators,
} from "../stage-story-helpers";
import Stage6Showcase from "./stage-6-showcase";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-6.json";
import nutsdaFixture from "../../../../../../data/DOCTOLIB/run/stage-6-nutsda.json";
import stage1aFixture from "../../../../../../data/DOCTOLIB/run/stage-1a.json";
import stage1bFixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage = STAGES.find((s) => s.id === "stage-6")!;
const stage1a = stage1aFixture as unknown as Stage1aData;
const stage1b = stage1bFixture as unknown as Stage1bData;
const data = { outcome: "success", ...fixture } as Stage6Data;
const nutsdaData = { outcome: "success", ...nutsdaFixture } as Stage6Data;
const defaultProjectionTreatment = data.intervention_results[0]?.treatment ?? null;
const latentDagDescription =
  "Upstream causal structure carried into treatment-effect interpretation.";

const storyTrace: LLMTrace = {
  model: "openrouter/anthropic/claude-sonnet-4",
  total_time_seconds: 4.2,
  usage: {
    input_tokens: 1765,
    output_tokens: 312,
    reasoning_tokens: 94,
  },
  messages: [
    {
      role: "system",
      content:
        "You are writing the opening commentary for Stage 6 of a causal state-space analysis. Comment on the treatment-effect results for a technical user.",
      tool_is_error: false,
    },
    {
      role: "user",
      content:
        "Comment the results of Stage 6 for the fitted model, note warnings, and mention available rung 2 and rung 3 follow-up simulations.",
      tool_is_error: false,
    },
    {
      role: "assistant",
      content:
        "Lipid burden is the strongest risk-increasing lever in the baseline ranking, while medication adherence and glycemic control are the strongest protective levers. The posterior intervals for the top effects stay clearly away from zero, so the direction of the ranking looks stable even before any follow-up simulation. You can now inspect the weighted DAG or ask for rung 2 and rung 3 simulations directly from this stage.",
      tool_is_error: false,
    },
  ],
};

const dataWithTrace = {
  ...data,
  llm_trace: storyTrace,
  final_summary:
    "Lipid burden is the strongest risk-increasing lever in the baseline ranking, while medication adherence and glycemic control are the strongest protective levers. The posterior intervals for the top effects stay clearly away from zero, so the direction of the ranking looks stable even before any follow-up simulation. You can now inspect the weighted DAG or ask for rung 2 and rung 3 simulations directly from this stage.",
} as Stage6Data;

const latentDagArgs = {
  dag: buildLatentDag(),
  dagTitle: "Latent DAG",
  dagDescription: latentDagDescription,
};

const completedArgs = {
  data: dataWithTrace,
  ...latentDagArgs,
  defaultSelectedTreatment: defaultProjectionTreatment,
};

const completedShellProps = {
  outcome: dataWithTrace.outcome,
  elapsedMs: 6_700,
  trace: storyTrace,
};

function buildLatentDag() {
  return (
    <CausalDag
      constructs={stage1a.latent_model.constructs}
      edges={stage1a.latent_model.edges}
      indicators={stage1b.causal_spec.measurement.indicators}
      height="600px"
    />
  );
}

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects",
  component: Stage6Showcase,
  decorators: stageStoryDecorators,
} satisfies Meta<typeof Stage6Showcase>;

export default meta;

export const Pending = createStageStatusStory(stage, "pending");

export const Running = createStageStatusStory(stage, "running");

export const Completed = createCompletedStageStory({
  name: "Completed (SVI / Laplace EM)",
  stage,
  args: completedArgs,
  ...completedShellProps,
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const OpenPanel = createOpenPanelStageStory({
  stage,
  args: completedArgs,
  ...completedShellProps,
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const CompletedNUTS = createCompletedStageStory({
  name: "Completed (NUTS / DA)",
  stage,
  args: {
    data: nutsdaData,
    ...latentDagArgs,
    defaultSelectedTreatment: nutsdaData.intervention_results[0]?.treatment ?? null,
  },
  outcome: nutsdaData.outcome,
  elapsedMs: 8_100,
  trace: storyTrace,
  renderContent: (args) => <Stage6Showcase {...args} />,
});

export const Failed = createStageStatusStory(stage, "failed");
