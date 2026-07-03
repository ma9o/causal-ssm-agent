import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { withContainer } from "@/components/story-decorators";
import { StageHeader } from "./stage-header";

const meta = {
  title: "Pipeline/StageHeader",
  component: StageHeader,
  decorators: [withContainer("max-w-3xl")],
  argTypes: {
    status: {
      control: "select",
      options: ["pending", "running", "completed", "failed"] satisfies StageRunStatus[],
    },
  },
} satisfies Meta<typeof StageHeader>;

export default meta;
type Story = StoryObj<typeof meta>;

// ---------------------------------------------------------------------------
// Badge colors by status
// ---------------------------------------------------------------------------

export const Pending: Story = {
  args: {
    number: "0",
    title: "Preprocess",
    status: "pending",
    context: "Parses raw data files and prepares them for downstream analysis.",
  },
};

export const Running: Story = {
  args: {
    number: "1a",
    title: "Latent Model",
    status: "running",
    context: "LLM is proposing a causal DAG...",
  },
};

export const Completed: Story = {
  args: {
    number: "2",
    title: "Data Extraction",
    status: "completed",
    context: "Dispatches worker LLMs to extract indicator observations from raw activity data.",
  },
};

export const RuntimeFailed: Story = {
  args: {
    number: "5b",
    title: "Inference & Diagnostics",
    status: "failed",
    context: "Fits the Bayesian model and runs convergence diagnostics.",
  },
};
