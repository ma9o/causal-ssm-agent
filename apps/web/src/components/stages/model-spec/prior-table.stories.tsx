import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { PriorProposal, Stage4Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { PriorTable } from "./prior-table";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-4.json";

const data = fixture as unknown as Stage4Data;
const priors = Object.values(data.priors).filter(Boolean) as PriorProposal[];
const parameters = data.model_spec.parameters;

const meta = {
  title: "Stages/ModelSpec/PriorTable",
  component: PriorTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof PriorTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithSearchContext: Story = {
  args: { priors, parameters },
};

export const WithoutSearchContext: Story = {
  args: { priors },
};
