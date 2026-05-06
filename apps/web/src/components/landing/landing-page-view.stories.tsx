import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { AccessibleWorkspaceList } from "@/lib/server/workspace-ownership";
import { AccessibleWorkspacesRail } from "@/components/pipeline/accessible-workspaces-rail";
import { LandingPageView } from "./landing-page-view";

const noop = () => {};

const anonymousWorkspaces: AccessibleWorkspaceList = {
  mode: "anonymous",
  workspaces: [
    {
      href: "/analysis/A8X2MN4Q1L9P",
      question:
        "How does my evening phone use affect sleep quality the following morning?",
      source: "session",
      workspaceId: "A8X2MN4Q1L9P",
    },
    {
      href: "/analysis/DEFAULT",
      question:
        "How does commute intensity affect stress and sleep in the DemoHealth fixture?",
      source: "shared",
      workspaceId: "DEFAULT",
    },
    {
      href: "/analysis/GOLDEN",
      question:
        "Golden fixture for smoke-testing the full pipeline and UI rendering.",
      source: "shared",
      workspaceId: "GOLDEN",
    },
  ],
};

const userWorkspaces: AccessibleWorkspaceList = {
  mode: "user",
  workspaces: [
    {
      href: "/analysis/U19K4P2Q8M7R",
      question:
        "Does time spent outdoors reduce next-day rumination and improve mood stability?",
      source: "user",
      workspaceId: "U19K4P2Q8M7R",
    },
    {
      href: "/analysis/U7F3D1C9X5TA",
      question:
        "What is the effect of caffeine timing on sleep onset latency and morning energy?",
      source: "user",
      workspaceId: "U7F3D1C9X5TA",
    },
    {
      href: "/analysis/DEFAULT",
      question:
        "How does commute intensity affect stress and sleep in the DemoHealth fixture?",
      source: "shared",
      workspaceId: "DEFAULT",
    },
  ],
};

const localWorkspaces: AccessibleWorkspaceList = {
  mode: "local",
  workspaces: [
    {
      href: "/analysis/local-adhd-pilot",
      question:
        "Local ADHD pilot workspace with merged EMA and wearable measurements.",
      source: "local",
      workspaceId: "local-adhd-pilot",
    },
    {
      href: "/analysis/local-sleep-study",
      question:
        "Local sleep study workspace with irregular actigraphy and survey exports.",
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

type StoryArgs = React.ComponentProps<typeof LandingPageView> & {
  railData?: AccessibleWorkspaceList;
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
      <AccessibleWorkspacesRail
        data={railData}
        error={railError ?? null}
        isLoading={railLoading ?? false}
      />
    </div>
  ),
  args: {
    access: {
      authScope: "anonymous",
      mode: "anonymous" as const,
      canRun: true as const,
      creditStatus: "available" as const,
    },
    noAccess: false,
    onSignOut: noop,
    onOpenRouterAuth: noop,
    question: "",
    onQuestionChange: noop,
    file: null,
    onFileSelect: noop,
    onFileRemove: noop,
    isSubmitting: false,
    submitDisabled: true,
    onSubmit: noop,
    error: null,
    railData: anonymousWorkspaces,
    railError: null,
    railLoading: false,
  },
  argTypes: {
    access: { control: "object" },
    noAccess: { control: "boolean" },
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

/** Default state: anonymous mode with shared credits available. */
export const Default: Story = {};

/** User has signed in with OpenRouter. */
export const SignedIn: Story = {
  args: {
    access: { authScope: "user:story-user", mode: "user", canRun: true },
    railData: userWorkspaces,
  },
};

/** Local development uses the server key directly and hides BYOK auth controls. */
export const LocalMode: Story = {
  args: {
    access: { authScope: "local", mode: "local", canRun: true },
    railData: localWorkspaces,
  },
};

/** No access — submit blocked, sign-in CTA prominent. */
export const NoAccess: Story = {
  args: {
    access: {
      authScope: "none:anonymous_exhausted",
      mode: "none",
      canRun: false,
      reason: "anonymous_exhausted",
    },
    noAccess: true,
  },
};

/** Both question and file filled — ready to submit. */
export const ReadyToSubmit: Story = {
  args: {
    question:
      "How does my daily screen time affect my sleep quality and mood?",
    file: { name: "google-takeout-2024.zip", size: 15_400_000 },
    submitDisabled: false,
  },
};

/** Analysis is being started — spinner active, submit disabled. */
export const Submitting: Story = {
  args: {
    question:
      "How does my daily screen time affect my sleep quality and mood?",
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

/** No workspaces exist yet for this user. */
export const EmptyWorkspaces: Story = {
  args: {
    railData: { mode: "anonymous", workspaces: [] },
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
    railError: "Failed to load accessible workspaces.",
  },
};
