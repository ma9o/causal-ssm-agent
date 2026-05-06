import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5bAuxGibbs } from "@/components/__fixtures__/inference-data";
import { EnergyChart } from "./energy-chart";

const meta = {
  title: "Charts/EnergyChart",
  component: EnergyChart,
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof EnergyChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { energy: stage5bAuxGibbs.mcmc_diagnostics!.energy! },
};
