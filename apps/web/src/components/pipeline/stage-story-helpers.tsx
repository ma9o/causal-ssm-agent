import type { StageRunStatus } from "@/lib/hooks/use-run-events";
import type { StageMeta } from "@nof1-causal-lab/api-types";
import type { Decorator } from "@storybook/nextjs-vite";
import type { ReactNode } from "react";
import { StoryStageLogView } from "./stage-story-log-stream";
import {
  StageStoryLayout,
  StageStoryTemplate,
  type StageStoryTemplateProps,
} from "./stage-story-template";

type StoryShellProps = Omit<StageStoryTemplateProps, "stage" | "status" | "children">;

export const stageStoryDecorators: Decorator[] = [
  (Story) => (
    <StageStoryLayout>
      <Story />
    </StageStoryLayout>
  ),
];

function getDefaultStoryLogView(stage: StageMeta, status: StageRunStatus): ReactNode | undefined {
  if (status === "pending") return undefined;

  return <StoryStageLogView storyId={`${stage.id}-${status}`} status={status} />;
}

type GeneratedStory<TArgs extends object = Record<string, never>> = {
  args?: TArgs;
  name?: string;
  render: ((storyArgs: TArgs) => ReactNode) | (() => ReactNode);
};

export function createStageStatusStory(
  stage: StageMeta,
  status: StageRunStatus,
  shellProps: StoryShellProps = {},
): GeneratedStory {
  const { logView = getDefaultStoryLogView(stage, status), ...restShellProps } = shellProps;
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
}: CompletedStageStoryConfig<TArgs>): GeneratedStory<TArgs> {
  const { logView = getDefaultStoryLogView(stage, "completed"), ...restShellProps } = shellProps;
  return {
    ...(name ? { name } : {}),
    args,
    render: (storyArgs: TArgs) => {
      const dynamicShellProps = renderShellProps?.(storyArgs) ?? {};
      const resolvedLogView = dynamicShellProps.logView ?? logView;

      return (
        <StageStoryTemplate
          stage={stage}
          status="completed"
          logView={resolvedLogView}
          {...restShellProps}
          {...dynamicShellProps}
        >
          {renderContent(storyArgs)}
        </StageStoryTemplate>
      );
    },
  };
}
