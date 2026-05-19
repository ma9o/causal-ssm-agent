import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { StageOutcome } from "@nof1-causal-lab/api-types";
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
    outcome: {
      control: "select",
      options: ["success", "warn", "fail"] satisfies StageOutcome[],
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

export const CompletedSuccess: Story = {
  args: {
    number: "2",
    title: "Data Extraction",
    status: "completed",
    outcome: "success",
    context: "Dispatches worker LLMs to extract indicator observations from raw activity data.",
  },
};

export const CompletedWarn: Story = {
  args: {
    number: "3",
    title: "Validation",
    status: "completed",
    outcome: "warn",
    context: "Some indicators had low extraction coverage.",
  },
};

export const CompletedFail: Story = {
  args: {
    number: "1b",
    title: "Measurement & Nonparametric Identification",
    status: "completed",
    outcome: "fail",
    context: "One or more treatment effects are not identifiable via do-calculus.",
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
