import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5bAuxGibbs } from "@/components/__fixtures__/inference-data";
import { PowerScalingScatter } from "./power-scaling-scatter";

const meta = {
  title: "Charts/PowerScalingScatter",
  component: PowerScalingScatter,
  decorators: [withContainer("max-w-md")],
} satisfies Meta<typeof PowerScalingScatter>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { results: stage5bAuxGibbs.power_scaling ?? [] },
};
