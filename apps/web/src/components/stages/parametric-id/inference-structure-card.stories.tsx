import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage4bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { InferenceStructureCard } from "./inference-structure-card";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-4b.json";

const data = fixture as unknown as Stage4bData;

const meta = {
  title: "Stages/ParametricId/InferenceStructureCard",
  component: InferenceStructureCard,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-3xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof InferenceStructureCard>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { inferenceStructure: data.inference_structure! },
};
