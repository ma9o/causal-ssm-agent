import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { MeasurementTable } from "./measurement-table";
import {
  likelihoodDiagnostics,
  likelihoods,
  priorPredictiveSamples,
} from "./__fixtures__/model-spec-fixtures";

const meta = {
  title: "Stages/ModelSpec/MeasurementTable",
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
