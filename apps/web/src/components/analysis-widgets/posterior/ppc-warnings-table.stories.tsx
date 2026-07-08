import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { posteriorAuxKalmanMCMC } from "@/components/__fixtures__/inference-data";
import { PPCWarningsTable } from "./ppc-warnings-table";

const ppc = posteriorAuxKalmanMCMC.ppc!;

const meta = {
  title: "Pipeline/Outputs/Posterior/PPCWarningsTable",
  component: PPCWarningsTable,
  decorators: [withContainer()],
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
