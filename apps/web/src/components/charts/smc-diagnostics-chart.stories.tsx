import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5b } from "@/components/__fixtures__/inference-data";
import { SMCDiagnosticsChart } from "./smc-diagnostics-chart";

const meta = {
  title: "Charts/SMCDiagnosticsChart",
  component: SMCDiagnosticsChart,
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof SMCDiagnosticsChart>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { diagnostics: stage5b.smc_diagnostics! },
};
