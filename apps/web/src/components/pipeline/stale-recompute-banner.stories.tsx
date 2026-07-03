import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { StaleRecomputeBannerView } from "./stale-recompute-banner";

const meta = {
  title: "Pipeline/StaleRecomputeBanner",
  component: StaleRecomputeBannerView,
  decorators: [withContainer("max-w-4xl")],
  args: {
    onRecompute: () => {},
  },
} satisfies Meta<typeof StaleRecomputeBannerView>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Stale: Story = {
  args: {
    staleStageCount: 3,
    recomputing: false,
  },
};

export const SingleStage: Story = {
  args: {
    staleStageCount: 1,
    recomputing: false,
  },
};

export const Recomputing: Story = {
  args: {
    staleStageCount: 3,
    recomputing: true,
  },
};

export const WithError: Story = {
  args: {
    staleStageCount: 2,
    recomputing: false,
    error: "Recompute failed",
  },
};
