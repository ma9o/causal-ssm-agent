import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { StageOutcome } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import { StageSection } from "./stage-section";

const meta = {
  title: "Pipeline/StageSection",
  component: StageSection,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
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
    outcome: {
      control: "select",
      options: ["success", "warn", "fail"] satisfies StageOutcome[],
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
    outcome: "success",
    context: "Parses raw data files and prepares them for downstream analysis.",
    elapsedMs: 4_320,
    children: (
      <div className="rounded-md border bg-muted/30 p-4 text-sm text-muted-foreground">
        Stage content would appear here.
      </div>
    ),
  },
};
