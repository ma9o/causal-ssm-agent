import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage4bData } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { InferenceStructureCard } from "./inference-structure-card";
import fixture from "../../../../../../data/GOLDEN/run/stage-4b.json";

const data = fixture as unknown as Stage4bData;

const meta = {
  title: "Stages/ParametricId/InferenceStructureCard",
  component: InferenceStructureCard,
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof InferenceStructureCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { inferenceStructure: data.inference_structure! },
};
