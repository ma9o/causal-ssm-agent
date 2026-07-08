import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { posterior } from "@/components/__fixtures__/inference-data";
import { PosteriorPairsChart } from "./posterior-pairs-chart";

const pairs = posterior.posterior_pairs ?? [];

const meta = {
  title: "Charts/PosteriorPairsChart",
  component: PosteriorPairsChart,
  decorators: [withContainer("max-w-sm")],
} satisfies Meta<typeof PosteriorPairsChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { pair: pairs[0] },
};
