import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { HardGateAlert } from "./hard-gate-alert";

const meta = {
  title: "UI/HardGateAlert",
  component: HardGateAlert,
  decorators: [
    (Story) => (
      <div className="max-w-3xl mx-auto p-4">
        <Story />
      </div>
    ),
  ],
} satisfies Meta<typeof HardGateAlert>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  render: () => (
    <HardGateAlert
      title="Identifiability Failed"
      explanation="None of the proposed treatment effects are nonparametrically identifiable from the current DAG structure. The pipeline cannot proceed without at least one identifiable causal path."
    />
  ),
};

export const WithChildren: Story = {
  render: () => (
    <HardGateAlert
      title="Validation Hard Failure"
      explanation="Critical data quality issues were detected that prevent model fitting."
    >
      <ul className="list-disc pl-5 text-sm">
        <li>Zero variance detected in 3 indicators</li>
        <li>More than 50% of timestamps are unparseable</li>
      </ul>
    </HardGateAlert>
  ),
};
