import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { TransitionRunStatus } from "@/lib/hooks/use-run-events";
import { withContainer } from "@/components/story-decorators";
import { OutputHeader } from "./output-header";

const meta = {
  title: "Pipeline/OutputHeader",
  component: OutputHeader,
  decorators: [withContainer("max-w-3xl")],
  argTypes: {
    status: {
      control: "select",
      options: ["pending", "running", "completed", "failed"] satisfies TransitionRunStatus[],
    },
  },
} satisfies Meta<typeof OutputHeader>;

export default meta;
type Story = StoryObj<typeof meta>;

// ---------------------------------------------------------------------------
// Badge colors by status
// ---------------------------------------------------------------------------

export const Pending: Story = {
  args: {
    title: "Preprocess",
    status: "pending",
    context: "Parses raw data files and prepares them for downstream analysis.",
  },
};

export const Running: Story = {
  args: {
    title: "Latent Structure",
    status: "running",
    context: "LLM is proposing a causal DAG...",
  },
};

export const Completed: Story = {
  args: {
    title: "Data Extraction",
    status: "completed",
    context: "Dispatches worker LLMs to extract indicator observations from raw activity data.",
  },
};

export const RuntimeFailed: Story = {
  args: {
    title: "Inference & Diagnostics",
    status: "failed",
    context: "Fits the Bayesian model and runs convergence diagnostics.",
  },
};

/** Completed output whose produced artifacts went stale after an upstream write. */
export const Stale: Story = {
  args: {
    title: "Measurement & Nonparametric Identification",
    status: "completed",
    context: "Maps latent constructs to observable indicators.",
    staleArtifactIds: ["causal_design", "identification_report"],
  },
};
