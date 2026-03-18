import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage4Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { PriorTable } from "./prior-table";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-4.json";

const data = fixture as Stage4Data;
const priors = data.priors;
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
  render: () => <PriorTable priors={priors} parameters={parameters} />,
};

export const WithoutSearchContext: Story = {
  render: () => <PriorTable priors={priors} />,
};
