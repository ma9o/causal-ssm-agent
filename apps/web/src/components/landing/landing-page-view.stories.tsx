import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { LandingPageView } from "./landing-page-view";

const noop = () => {};

const meta = {
  title: "Landing/LandingPageView",
  component: LandingPageView,
  args: {
    access: {
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
  },
  argTypes: {
    access: { control: "object" },
    noAccess: { control: "boolean" },
    question: { control: "text" },
    isSubmitting: { control: "boolean" },
    submitDisabled: { control: "boolean" },
    error: { control: "text" },
  },
} satisfies Meta<typeof LandingPageView>;

export default meta;
type Story = StoryObj<typeof meta>;

/** Default state: anonymous mode with shared credits available. */
export const Default: Story = {};

/** User has signed in with OpenRouter. */
export const SignedIn: Story = {
  args: {
    access: { mode: "user", canRun: true },
  },
};

/** Local development uses the server key directly and hides BYOK auth controls. */
export const LocalMode: Story = {
  args: {
    access: { mode: "local", canRun: true },
  },
};

/** No access — submit blocked, sign-in CTA prominent. */
export const NoAccess: Story = {
  args: {
    access: { mode: "none", canRun: false, reason: "anonymous_exhausted" },
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
