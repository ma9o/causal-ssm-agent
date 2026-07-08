import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { TransitionRunStatus } from "@/lib/hooks/use-run-events";
import { withContainer } from "@/components/story-decorators";
import { OutputSection } from "./output-section";

const meta = {
  title: "Pipeline/OutputSection",
  component: OutputSection,
  decorators: [withContainer("max-w-3xl")],
  args: {
    title: "Example Artifact",
    status: "pending",
    context: "An artifact description for demonstration purposes.",
  },
  argTypes: {
    status: {
      control: "select",
      options: ["pending", "running", "completed", "failed"] satisfies TransitionRunStatus[],
    },
  },
} satisfies Meta<typeof OutputSection>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Playground: Story = {
  args: {
    title: "Preprocess",
    status: "completed",
    context: "Parses raw data files and prepares them for downstream analysis.",
    elapsedMs: 4_320,
    children: (
      <div className="rounded-md border bg-muted/30 p-4 text-sm text-muted-foreground">
        Output content would appear here.
      </div>
    ),
  },
};

/** Running output — loading hint plus skeleton placeholders. */
export const Running: Story = {
  args: {
    title: "Latent Structure",
    status: "running",
    context: "Generating latent structure from user question.",
    loadingHint: "Querying the LLM for causal structure…",
  },
};

/** Failed output — the raised error detail stays visible. */
export const FailedWithError: Story = {
  args: {
    title: "Latent Structure",
    status: "failed",
    context: "Generates the latent structure from the user question.",
    errorMessage: "SchemaValidationError: latent_structure payload failed validation",
  },
};
