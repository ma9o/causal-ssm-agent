import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { CausalDag } from "./causal-dag";
import {
  blockingEdges,
  constructs,
  edges,
  indicators,
  nodeStatuses,
} from "./__fixtures__/dag-base-fixtures";

const meta = {
  title: "DAG/CausalDag",
  component: CausalDag,
  decorators: [withContainer("max-w-3xl")],
} satisfies Meta<typeof CausalDag>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { constructs, edges },
};

export const WithIndicators: Story = {
  args: { constructs, edges, indicators },
};

export const WithNodeStatuses: Story = {
  args: { constructs, edges, indicators, nodeStatuses },
};

export const WithBlockingEdge: Story = {
  args: { constructs, edges, indicators, nodeStatuses, blockingEdges },
};

export const Empty: Story = {
  args: { constructs: [], edges: [] },
};
