import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { StageMeta } from "@causal-ssm/api-types";
import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { ReactNode } from "react";
import { StageLogView } from "./stage-log-viewer";
import { createStageStoryLogs } from "./stage-story-log-fixture";
import {
  StageStoryLayout,
  StageStoryTemplate,
  type StageStoryTemplateProps,
} from "./stage-story-template";

type StoryShellProps = Omit<StageStoryTemplateProps, "stage" | "status" | "children">;

export const stageStoryDecorators: NonNullable<Meta<Record<string, never>>["decorators"]> = [
  (Story) => (
    <StageStoryLayout>
      <Story />
    </StageStoryLayout>
  ),
];

function getDefaultStoryLogView(status: StageRunStatus): ReactNode | undefined {
  if (status === "pending") return undefined;

  return (
    <StageLogView
      logs={createStageStoryLogs()}
      status={status}
      bootstrapStatus="success"
      connectionState={status === "running" ? "streaming" : "idle"}
    />
  );
}

export function createStageStatusStory(
  stage: StageMeta,
  status: StageRunStatus,
  shellProps: StoryShellProps = {},
): StoryObj {
  const { logView = getDefaultStoryLogView(status), ...restShellProps } = shellProps;
  return {
    render: () => (
      <StageStoryTemplate stage={stage} status={status} logView={logView} {...restShellProps} />
    ),
  };
}

type CompletedStageStoryConfig<TArgs extends object> = StoryShellProps & {
  stage: StageMeta;
  args: TArgs;
  name?: string;
  renderContent: (args: TArgs) => ReactNode;
  renderShellProps?: (args: TArgs) => Partial<StoryShellProps>;
};

export function createCompletedStageStory<TArgs extends object>({
  stage,
  args,
  name,
  renderContent,
  renderShellProps,
  ...shellProps
}: CompletedStageStoryConfig<TArgs>): StoryObj<TArgs> {
  const { logView = getDefaultStoryLogView("completed"), ...restShellProps } = shellProps;
  return {
    ...(name ? { name } : {}),
    args,
    render: (storyArgs) => {
      const dynamicShellProps = renderShellProps?.(storyArgs as TArgs) ?? {};
      const resolvedLogView = dynamicShellProps.logView ?? logView;

      return (
        <StageStoryTemplate
          stage={stage}
          status="completed"
          logView={resolvedLogView}
          {...restShellProps}
          {...dynamicShellProps}
        >
          {renderContent(storyArgs as TArgs)}
        </StageStoryTemplate>
      );
    },
  };
}
