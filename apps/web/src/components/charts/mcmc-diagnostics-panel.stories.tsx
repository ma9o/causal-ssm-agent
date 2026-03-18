import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { MCMCDiagnosticsPanel } from "./mcmc-diagnostics-panel";
import fixture from "../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";

const data = fixture as Stage5bData;

const meta = {
  title: "Charts/MCMCDiagnosticsPanel",
  component: MCMCDiagnosticsPanel,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof MCMCDiagnosticsPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => (
    <MCMCDiagnosticsPanel diagnostics={data.mcmc_diagnostics!} />
  ),
};
