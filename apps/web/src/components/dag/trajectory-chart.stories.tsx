import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { counterfactualResult, interventionResult } from "./__fixtures__/intervention-dag-fixture";
import { TrajectoryChart } from "./trajectory-chart";

const meta = {
  title: "Pipeline/Stages/6 – Treatment Effects/Trajectory Chart",
  component: TrajectoryChart,
  decorators: [withContainer("max-w-2xl")],
} satisfies Meta<typeof TrajectoryChart>;

export default meta;

type Story = StoryObj<typeof TrajectoryChart>;

export const EffectMetric: Story = {
  args: { result: interventionResult, metric: "effect" },
};

export const ActionPath: Story = {
  args: { result: interventionResult, metric: "action" },
};

export const Reference: Story = {
  args: { result: interventionResult, metric: "reference" },
};

export const Counterfactual: Story = {
  args: { result: counterfactualResult, metric: "effect" },
};
