import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { PosteriorData } from "@nof1-causal-lab/api-types";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import PosteriorView from "./posterior-view";
import { demoPosterior } from "../../__fixtures__/demo-artifacts";

const output = TRANSITIONS.find((s) => s.id === "posterior")!;
const data = demoPosterior as PosteriorData;
const auxKalmanMCMCData = demoPosterior as PosteriorData;

const meta = {
  title: "Pipeline/Outputs/Posterior/Panel",
  component: PosteriorView,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof PosteriorView>;

export default meta;

export const Pending = createOutputStatusStory(output, "pending");

export const Running = createOutputStatusStory(output, "running");

export const CompletedMAP = createCompletedOutputStory({
  name: "Completed (MAP)",
  output,
  args: { data, workspaceId: "demo-user" },
  elapsedMs: 124_500,
  renderContent: (args) => <PosteriorView {...args} />,
});

export const CompletedAuxKalmanMCMC = createCompletedOutputStory({
  name: "Completed (Auxiliary Kalman MCMC)",
  output,
  args: { data: auxKalmanMCMCData, workspaceId: "demo-user" },
  elapsedMs: 342_000,
  renderContent: (args) => <PosteriorView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
