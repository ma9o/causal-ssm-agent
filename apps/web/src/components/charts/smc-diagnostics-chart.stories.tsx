import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SMCDiagnosticsChart } from "./smc-diagnostics-chart";
import fixture from "../../../../../data/DOCTOLIB/run/stage-5b.json";

const data = fixture as Stage5bData;

const meta = {
  title: "Charts/SMCDiagnosticsChart",
  component: SMCDiagnosticsChart,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof SMCDiagnosticsChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => (
    <SMCDiagnosticsChart diagnostics={data.smc_diagnostics!} />
  ),
};
