import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { PriorTable } from "./prior-table";
import { priors, parameters } from "./__fixtures__/model-spec-fixtures";

const meta = {
  title: "Pipeline/Stages/4 – Model Specification/PriorTable",
  component: PriorTable,
  decorators: [withContainer()],
} satisfies Meta<typeof PriorTable>;

export default meta;
type Story = StoryObj<typeof meta>;

export const WithSearchContext: Story = {
  args: { priors, parameters },
};

export const WithoutSearchContext: Story = {
  args: { priors },
};
