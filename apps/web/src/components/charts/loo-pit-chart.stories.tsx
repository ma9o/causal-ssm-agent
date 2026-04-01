import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5b } from "@/components/__fixtures__/inference-data";
import { LOOPITChart } from "./loo-pit-chart";

const meta = {
  title: "Charts/LOOPITChart",
  component: LOOPITChart,
  decorators: [withContainer("max-w-md")],
} satisfies Meta<typeof LOOPITChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { loo: stage5b.loo_diagnostics! },
};
