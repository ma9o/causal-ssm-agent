import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { ObsModelTable } from "./obs-model-table";
import {
  likelihoods,
  parameters,
  priors,
  indicatorConstructMap,
} from "./__fixtures__/model-spec-fixtures";

const meta = {
  title: "Stages/ModelSpec/ObsModelTable",
  component: ObsModelTable,
  decorators: [withContainer()],
} satisfies Meta<typeof ObsModelTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithIndicatorMapping: Story = {
  args: { likelihoods, parameters, priors, indicatorConstructMap },
};

export const WithoutIndicatorMapping: Story = {
  args: { likelihoods, parameters, priors },
};
