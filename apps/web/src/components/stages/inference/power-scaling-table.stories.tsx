import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5bData } from "@causal-ssm/api-types";
import { withContainer } from "@/components/story-decorators";
import { stage5bNutsda } from "@/components/__fixtures__/inference-data";
import { PowerScalingTable } from "./power-scaling-table";

const meta = {
  title: "Stages/Inference/PowerScalingTable",
  component: PowerScalingTable,
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof PowerScalingTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithPSIS: Story = {
  args: { results: stage5bNutsda.power_scaling ?? [] },
};

export const WithoutPSIS: Story = {
  args: {
    results: (stage5bNutsda.power_scaling ?? []).map(({ psis_k_hat, ...rest }) => rest) as Stage5bData["power_scaling"] & {},
  },
};
