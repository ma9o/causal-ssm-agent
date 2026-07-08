import type { Meta } from "@storybook/nextjs-vite";
import { TRANSITIONS } from "@nof1-causal-lab/api-types";
import type { LatentStructureData } from "@nof1-causal-lab/api-types";
import {
  createCompletedOutputStory,
  createOutputStatusStory,
  outputStoryDecorators,
} from "../output-story-helpers";
import LatentStructureView from "./latent-structure-view";
import fixture from "../../__fixtures__/demo-run/latent_structure.json";

const output = TRANSITIONS.find((s) => s.id === "latent_structure")!;
const data = fixture as unknown as LatentStructureData;

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
  trace: data.llm_trace ?? undefined,
  renderContent: (args) => <LatentStructureView {...args} />,
});

export const OpenPanel = createCompletedOutputStory({
  output,
  args: { data },
  elapsedMs: 12_450,
  trace: data.llm_trace ?? undefined,
  defaultPanelOpen: true,
  renderContent: (args) => <LatentStructureView {...args} />,
});

export const Failed = createOutputStatusStory(output, "failed");
