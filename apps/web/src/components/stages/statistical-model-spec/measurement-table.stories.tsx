import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { MeasurementTable } from "./measurement-table";
import {
  likelihoodDiagnostics,
  likelihoods,
  priorPredictiveSamples,
} from "./__fixtures__/statistical-model-spec-fixtures";

const meta = {
  title: "Pipeline/Stages/4 – Statistical Model Specification/MeasurementTable",
  component: MeasurementTable,
  decorators: [withContainer()],
} satisfies Meta<typeof MeasurementTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { likelihoods, diagnostics: likelihoodDiagnostics },
};

export const WithPriorPredictive: Story = {
  args: { likelihoods, diagnostics: likelihoodDiagnostics, priorPredictiveSamples },
};
