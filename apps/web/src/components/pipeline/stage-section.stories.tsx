import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { withContainer } from "@/components/story-decorators";
import { StageSection } from "./stage-section";

const meta = {
  title: "Pipeline/StageSection",
  component: StageSection,
  decorators: [withContainer("max-w-3xl")],
  args: {
    number: "0",
    title: "Example Stage",
    status: "pending",
    context: "A stage description for demonstration purposes.",
  },
  argTypes: {
    status: {
      control: "select",
      options: ["pending", "running", "completed", "failed"] satisfies StageRunStatus[],
    },
  },
} satisfies Meta<typeof StageSection>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Playground: Story = {
  args: {
    number: "0",
    title: "Preprocess",
    status: "completed",
    context: "Parses raw data files and prepares them for downstream analysis.",
    elapsedMs: 4_320,
    children: (
      <div className="rounded-md border bg-muted/30 p-4 text-sm text-muted-foreground">
        Stage content would appear here.
      </div>
    ),
  },
};

/** Running stage — loading hint plus skeleton placeholders. */
export const Running: Story = {
  args: {
    number: "1a",
    title: "Latent Structure",
    status: "running",
    context: "Generating latent structure from user question.",
    loadingHint: "Querying the LLM for causal structure…",
  },
};

/** Failed stage — the raised error detail stays visible. */
export const FailedWithError: Story = {
  args: {
    number: "1a",
    title: "Latent Structure",
    status: "failed",
    context: "Generates the latent structure from the user question.",
    errorMessage: "SchemaValidationError: latent_structure payload failed validation",
  },
};
