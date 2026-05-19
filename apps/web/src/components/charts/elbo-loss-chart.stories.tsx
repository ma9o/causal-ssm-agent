import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5b } from "@/components/__fixtures__/inference-data";
import { ELBOLossChart } from "./elbo-loss-chart";

const meta = {
  title: "Charts/ELBOLossChart",
  component: ELBOLossChart,
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof ELBOLossChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { diagnostics: stage5b.svi_diagnostics! },
};
