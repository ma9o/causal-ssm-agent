import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import type { AccessibleWorkspaceList } from "@/lib/server/workspace-ownership";
import { AccessibleWorkspacesRail } from "./accessible-workspaces-rail";

const anonymousData: AccessibleWorkspaceList = {
  mode: "anonymous",
  workspaces: [
    {
      href: "/analysis/A8X2MN4Q1L9P",
      question: "How does my evening phone use affect sleep quality the following morning?",
      source: "session",
      workspaceId: "A8X2MN4Q1L9P",
    },
    {
      href: "/analysis/DEFAULT",
      question: "How does commute intensity affect stress and sleep in the Doctolib fixture?",
      source: "shared",
      workspaceId: "DEFAULT",
    },
    {
      href: "/analysis/GOLDEN",
      question: "Golden fixture for smoke-testing the full pipeline and UI rendering.",
      source: "shared",
      workspaceId: "GOLDEN",
    },
  ],
};

const userData: AccessibleWorkspaceList = {
  mode: "user",
  workspaces: [
    {
      href: "/analysis/U19K4P2Q8M7R",
      question: "Does time spent outdoors reduce next-day rumination and improve mood stability?",
      source: "user",
      workspaceId: "U19K4P2Q8M7R",
    },
    {
      href: "/analysis/U7F3D1C9X5TA",
      question: "What is the effect of caffeine timing on sleep onset latency and morning energy?",
      source: "user",
      workspaceId: "U7F3D1C9X5TA",
    },
    {
      href: "/analysis/DEFAULT",
      question: "How does commute intensity affect stress and sleep in the Doctolib fixture?",
      source: "shared",
      workspaceId: "DEFAULT",
    },
  ],
};

const localData: AccessibleWorkspaceList = {
  mode: "local",
  workspaces: [
    {
      href: "/analysis/local-adhd-pilot",
      question: "Local ADHD pilot workspace with merged EMA and wearable measurements.",
      source: "local",
      workspaceId: "local-adhd-pilot",
    },
    {
      href: "/analysis/local-sleep-study",
      question: "Local sleep study workspace with irregular actigraphy and survey exports.",
      source: "local",
      workspaceId: "local-sleep-study",
    },
    {
      href: "/analysis/local-medication-trial",
      question: null,
      source: "local",
      workspaceId: "local-medication-trial",
    },
  ],
};

const meta = {
  title: "Pipeline/AccessibleWorkspacesRail",
  component: AccessibleWorkspacesRail,
  decorators: [withContainer("max-w-[26rem]")],
  args: {
    data: anonymousData,
    error: null,
    isLoading: false,
  },
  argTypes: {
    data: { control: "object" },
    error: { control: "text" },
    isLoading: { control: "boolean" },
  },
} satisfies Meta<typeof AccessibleWorkspacesRail>;

export default meta;
type Story = StoryObj<typeof meta>;

export const AnonymousMode: Story = {};

export const UserMode: Story = {
  args: {
    data: userData,
  },
};

export const LocalMode: Story = {
  args: {
    data: localData,
  },
};

export const EmptyState: Story = {
  args: {
    data: {
      mode: "anonymous",
      workspaces: [],
    },
  },
};

export const Loading: Story = {
  args: {
    data: undefined,
    isLoading: true,
  },
};

export const ErrorState: Story = {
  args: {
    data: undefined,
    error: "Failed to load accessible workspaces.",
  },
};
