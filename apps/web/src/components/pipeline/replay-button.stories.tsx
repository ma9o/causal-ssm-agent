import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { ReplayButton } from "./replay-button";

const meta = {
  title: "Pipeline/ReplayButton",
  component: ReplayButton,
  decorators: [
    (Story) => (
      <div className="max-w-md mx-auto p-4">
        <Story />
      </div>
    ),
  ],
  argTypes: {
    stageId: {
      control: "select",
      options: ["stage-1a", "stage-1b", "stage-4", "stage-0", "stage-2", "stage-3"],
    },
  },
} satisfies Meta<typeof ReplayButton>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Idle state for an interactive stage. */
export const Default: Story = {
  args: {
    userId: "demo-user",
    stageId: "stage-1a",
  },
};

/** Each interactive stage renders identically – just varies the stageId. */
export const Stage1b: Story = {
  args: {
    userId: "demo-user",
    stageId: "stage-1b",
  },
};

export const Stage4: Story = {
  args: {
    userId: "demo-user",
    stageId: "stage-4",
  },
};

/** Non-interactive stages render nothing (component returns null). */
export const NonInteractiveStage: Story = {
  args: {
    userId: "demo-user",
    stageId: "stage-0",
  },
};

/** Simulates the loading state by clicking the button with fetch mocked to never resolve. */
export const Replaying: Story = {
  args: {
    userId: "demo-user",
    stageId: "stage-1a",
  },
  play: async ({ canvasElement }) => {
    // Mock fetch to hang indefinitely so loading state persists visually
    window.fetch = (() => new Promise(() => {})) as typeof window.fetch;
    canvasElement.querySelector("button")?.click();
  },
};
