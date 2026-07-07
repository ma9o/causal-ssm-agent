import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage1aData } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { EdgeList } from "./edge-list";
import fixture from "../../__fixtures__/demo-run/stage-1a.json";

const data = fixture as unknown as Stage1aData;
const edges = data.latent_structure.edges;

const meta = {
  title: "Pipeline/Stages/1a – Latent Structure/EdgeList",
  component: EdgeList,
  decorators: [withContainer("max-w-md")],
} satisfies Meta<typeof EdgeList>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { edges },
};

export const Empty: Story = {
  args: { edges: [] },
};
