import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5bAuxKalmanMCMC } from "@/components/__fixtures__/inference-data";
import { MCMCDiagnosticsPanel } from "./mcmc-diagnostics-panel";

const meta = {
  title: "Charts/MCMCDiagnosticsPanel",
  component: MCMCDiagnosticsPanel,
  decorators: [withContainer()],
} satisfies Meta<typeof MCMCDiagnosticsPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { diagnostics: stage5bAuxKalmanMCMC.mcmc_diagnostics! },
};
