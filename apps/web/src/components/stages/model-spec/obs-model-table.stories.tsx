import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { PriorProposal, Stage1bData, Stage4Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ObsModelTable } from "./obs-model-table";
import stage4Fixture from "../../../../../../data/DOCTOLIB/run/stage-4.json";
import stage1bFixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage4 = stage4Fixture as unknown as Stage4Data;
const stage1b = stage1bFixture as unknown as Stage1bData;

const likelihoods = stage4.model_spec.likelihoods;
const parameters = stage4.model_spec.parameters;
const priors = Object.values(stage4.priors).filter(Boolean) as PriorProposal[];

const indicatorConstructMap: Record<string, string> = {};
for (const ind of stage1b.causal_spec.measurement.indicators) {
  indicatorConstructMap[ind.name] = ind.construct_name;
}

const meta = {
  title: "Stages/ModelSpec/ObsModelTable",
  component: ObsModelTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof ObsModelTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithIndicatorMapping: Story = {
  args: { likelihoods, parameters, priors, indicatorConstructMap },
};

export const WithoutIndicatorMapping: Story = {
  args: { likelihoods, parameters, priors },
};
