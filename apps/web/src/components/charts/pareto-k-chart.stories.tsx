import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { posterior } from "@/components/__fixtures__/inference-data";
import { ParetoKChart } from "./pareto-k-chart";

const meta = {
  title: "Charts/ParetoKChart",
  component: ParetoKChart,
  decorators: [withContainer("max-w-md")],
} satisfies Meta<typeof ParetoKChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { loo: posterior.loo_diagnostics! },
};
