import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage2Data, Stage4Data } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { MeasurementTable } from "./measurement-table";
import stage4Fixture from "../../../../../../data/DOCTOLIB/run/stage-4.json";
import stage2Fixture from "../../../../../../data/DOCTOLIB/run/stage-2.json";

const stage4 = stage4Fixture as unknown as Stage4Data;
const stage2 = stage2Fixture as unknown as Stage2Data;

const likelihoods = stage4.model_spec.likelihoods;
const extractions = stage2.combined_extractions_sample;
const priorPredictiveSamples = stage4.prior_predictive_samples as Record<string, number[]> | undefined;

const meta = {
  title: "Stages/ModelSpec/MeasurementTable",
  component: MeasurementTable,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof MeasurementTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { likelihoods, extractions },
};

export const WithPriorPredictive: Story = {
  args: { likelihoods, extractions, priorPredictiveSamples },
};
