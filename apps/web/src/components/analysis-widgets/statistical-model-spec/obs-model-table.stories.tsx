import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { ObsModelTable } from "./obs-model-table";
import {
  indicators,
  likelihoods,
  parameters,
  priors,
} from "./__fixtures__/statistical-model-spec-fixtures";

const meta = {
  title: "Pipeline/Outputs/Statistical Model Spec/ObsModelTable",
  component: ObsModelTable,
  decorators: [withContainer()],
} satisfies Meta<typeof ObsModelTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithIndicators: Story = {
  args: { likelihoods, parameters, priors, indicators },
};

export const WithoutIndicators: Story = {
  args: { likelihoods, parameters, priors },
};
