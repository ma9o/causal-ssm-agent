import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage1bData, Stage4Data } from "@causal-ssm/api-types";
import { collectStage4Priors } from "@/lib/stage4-data";
import { TooltipProvider } from "@/components/ui/tooltip";
import { SSMEquationDisplay } from "./ssm-equation-display";
import stage4Fixture from "../../../../../../data/DOCTOLIB/run/stage-4.json";
import stage1bFixture from "../../../../../../data/DOCTOLIB/run/stage-1b.json";

const stage4 = stage4Fixture as unknown as Stage4Data;
const stage1b = stage1bFixture as unknown as Stage1bData;

const likelihoods = stage4.model_spec.likelihoods;
const parameters = stage4.model_spec.parameters;
const priors = collectStage4Priors(stage4);

const indicatorConstructMap: Record<string, string> = {};
for (const ind of stage1b.causal_spec.measurement.indicators) {
  indicatorConstructMap[ind.name] = ind.construct_name;
}

const meta = {
  title: "Stages/ModelSpec/SSMEquationDisplay",
  component: SSMEquationDisplay,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof SSMEquationDisplay>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { likelihoods, parameters, priors, indicatorConstructMap },
};

export const WithoutIndicatorMap: Story = {
  args: { likelihoods, parameters, priors },
};
