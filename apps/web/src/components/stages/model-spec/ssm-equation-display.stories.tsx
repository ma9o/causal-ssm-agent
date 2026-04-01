import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { SSMEquationDisplay } from "./ssm-equation-display";
import {
  likelihoods,
  parameters,
  priors,
  indicatorConstructMap,
} from "./__fixtures__/model-spec-fixtures";

const meta = {
  title: "Stages/ModelSpec/SSMEquationDisplay",
  component: SSMEquationDisplay,
  decorators: [withContainer()],
} satisfies Meta<typeof SSMEquationDisplay>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { likelihoods, parameters, priors, indicatorConstructMap },
};

export const WithoutIndicatorMap: Story = {
  args: { likelihoods, parameters, priors },
};
