import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage2Data } from "@causal-ssm/api-types";
import { withContainer } from "@/components/story-decorators";
import { MeasurementTable } from "./measurement-table";
import { likelihoods, priorPredictiveSamples } from "./__fixtures__/model-spec-fixtures";
import stage2Fixture from "../../../../../../data/DOCTOLIB/run/stage-2.json";

const stage2 = stage2Fixture as unknown as Stage2Data;
const extractions = stage2.combined_extractions_sample;

const meta = {
  title: "Stages/ModelSpec/MeasurementTable",
  component: MeasurementTable,
  decorators: [withContainer()],
} satisfies Meta<typeof MeasurementTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { likelihoods, extractions },
};

export const WithPriorPredictive: Story = {
  args: { likelihoods, extractions, priorPredictiveSamples },
};
