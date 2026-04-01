import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5b } from "@/components/__fixtures__/inference-data";
import { PosteriorDensityChart } from "./posterior-density-chart";

const marginals = stage5b.posterior_marginals ?? [];

const meta = {
  title: "Charts/PosteriorDensityChart",
  component: PosteriorDensityChart,
  decorators: [withContainer("max-w-sm")],
} satisfies Meta<typeof PosteriorDensityChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { marginal: marginals[0] },
};
