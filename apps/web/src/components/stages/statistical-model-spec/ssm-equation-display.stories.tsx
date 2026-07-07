import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { SSMEquationDisplay } from "./ssm-equation-display";
import {
  indicators,
  likelihoods,
  parameters,
  priors,
} from "./__fixtures__/statistical-model-spec-fixtures";

const meta = {
  title: "Pipeline/Stages/4 – Statistical Model Specification/SSMEquationDisplay",
  component: SSMEquationDisplay,
  decorators: [withContainer()],
} satisfies Meta<typeof SSMEquationDisplay>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { likelihoods, parameters, priors, indicators },
};

export const WithoutIndicators: Story = {
  args: { likelihoods, parameters, priors },
};
