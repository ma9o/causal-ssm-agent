import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { WorkspaceList } from "@/lib/server/workspaces";
import { WorkspacesRail } from "@/components/pipeline/workspaces-rail";
import { LandingPageView } from "./landing-page-view";

const noop = () => {};

const localWorkspaces: WorkspaceList = {
  workspaces: [
    {
      href: "/analysis/local-adhd-pilot",
      question: "Local ADHD pilot workspace with merged EMA and wearable measurements.",
      workspaceId: "local-adhd-pilot",
    },
    {
      href: "/analysis/local-sleep-study",
      question: "Local sleep study workspace with irregular actigraphy and survey exports.",
      workspaceId: "local-sleep-study",
    },
    {
      href: "/analysis/local-medication-trial",
      question: null,
      workspaceId: "local-medication-trial",
    },
  ],
};

type StoryArgs = React.ComponentProps<typeof LandingPageView> & {
  railData?: WorkspaceList;
  railError?: string | null;
  railLoading?: boolean;
};

const meta = {
  title: "Landing/LandingPageView",
  component: LandingPageView,
  render: ({ railData, railError, railLoading, ...args }) => (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 py-6 sm:px-6 xl:grid xl:grid-cols-[1fr_auto_1fr] xl:items-center xl:gap-6">
      <LandingPageView {...args} />
      <div className="hidden xl:block w-px h-2/3 bg-border" />
      <WorkspacesRail data={railData} error={railError ?? null} isLoading={railLoading ?? false} />
    </div>
  ),
  args: {
    movesEnabled: true,
    question: "",
    onQuestionChange: noop,
    file: null,
    onFileSelect: noop,
    onFileRemove: noop,
    isSubmitting: false,
    submitDisabled: true,
    onSubmit: noop,
    error: null,
    railData: localWorkspaces,
    railError: null,
    railLoading: false,
  },
  argTypes: {
    movesEnabled: { control: "boolean" },
    question: { control: "text" },
    isSubmitting: { control: "boolean" },
    submitDisabled: { control: "boolean" },
    error: { control: "text" },
    railData: { control: "object" },
    railError: { control: "text" },
    railLoading: { control: "boolean" },
  },
} satisfies Meta<StoryArgs>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Default state: full facade, run form available. */
export const Default: Story = {};

/** Hosted read-only viewer: no run form, published workspaces only. */
export const ReadOnlyViewer: Story = {
  args: {
    movesEnabled: false,
  },
};

/** Both question and file filled — ready to submit. */
export const ReadyToSubmit: Story = {
  args: {
    question: "How does my daily screen time affect my sleep quality and mood?",
    file: { name: "google-takeout-2024.zip", size: 15_400_000 },
    submitDisabled: false,
  },
};

/** Analysis is being started — spinner active, submit disabled. */
export const Submitting: Story = {
  args: {
    question: "How does my daily screen time affect my sleep quality and mood?",
    file: { name: "google-takeout-2024.zip", size: 15_400_000 },
    isSubmitting: true,
    submitDisabled: true,
  },
};

/** Validation error shown below the form. */
export const WithError: Story = {
  args: {
    error: "Please enter a research question.",
  },
};

/** No workspaces exist yet. */
export const EmptyWorkspaces: Story = {
  args: {
    railData: { workspaces: [] },
  },
};

/** Workspaces rail is loading. */
export const WorkspacesLoading: Story = {
  args: {
    railData: undefined,
    railLoading: true,
  },
};

/** Workspaces rail failed to load. */
export const WorkspacesError: Story = {
  args: {
    railData: undefined,
    railError: "Failed to load workspaces.",
  },
};
