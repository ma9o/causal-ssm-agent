import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { LatentStructureData } from "@nof1-causal-lab/api-types";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import LatentStructureView from "./latent-structure-view";
import { demoLatentStructure } from "../../__fixtures__/demo-artifacts";
import { demoTraces } from "../../__fixtures__/demo-traces";

const output = TRANSITIONS.find((s) => s.id === "latent_structure")!;
const data = demoLatentStructure as LatentStructureData;

const meta = {
  title: "Pipeline/Outputs/Latent Structure/Panel",
  component: LatentStructureView,
  decorators: outputStoryDecorators,
} satisfies Meta<typeof LatentStructureView>;

export default meta;

export const Pending = createOutputStatusStory(output, "pending");

export const Running = createOutputStatusStory(output, "running");

export const Completed = createCompletedOutputStory({
  output,
  args: { data },
  elapsedMs: 12_450,
  trace: demoTraces.latent_structure,
  renderContent: (args) => <LatentStructureView {...args} />,
});

export const OpenPanel = createCompletedOutputStory({
  output,
  args: { data },
  elapsedMs: 12_450,
  defaultPanelOpen: true,
  trace: demoTraces.latent_structure,
  renderContent: (args) => <LatentStructureView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
