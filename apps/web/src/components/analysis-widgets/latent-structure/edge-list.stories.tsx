import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { LatentStructureData } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { EdgeList } from "./edge-list";
import { demoLatentStructure } from "../../__fixtures__/demo-artifacts";

const data = demoLatentStructure as LatentStructureData;
const edges = data.latent_structure.edges;

const meta = {
  title: "Pipeline/Outputs/Latent Structure/EdgeList",
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
