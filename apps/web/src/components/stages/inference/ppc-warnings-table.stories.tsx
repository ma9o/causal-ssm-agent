import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { PPCWarningsTable } from "./ppc-warnings-table";
import nutsdaFixture from "../../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";

const data = nutsdaFixture as Stage5bData;
const ppc = data.ppc!;

const meta = {
  title: "Stages/Inference/PPCWarningsTable",
  component: PPCWarningsTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof PPCWarningsTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    warnings: ppc.per_variable_warnings,
    testStats: ppc.test_stats ?? [],
    overlays: ppc.overlays ?? [],
  },
};

export const WarningsOnly: Story = {
  args: {
    warnings: ppc.per_variable_warnings,
    testStats: [],
    overlays: [],
  },
};
