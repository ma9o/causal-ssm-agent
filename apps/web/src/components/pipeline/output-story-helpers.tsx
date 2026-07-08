import type { TransitionRunStatus } from "@/lib/hooks/use-run-events";
import type { TransitionMeta } from "@nof1-causal-lab/api-types";
import type { Decorator } from "@storybook/nextjs-vite";
import type { ReactNode } from "react";
import {
  OutputStoryLayout,
  OutputStoryTemplate,
  type OutputStoryTemplateProps,
} from "./output-story-template";

type StoryShellProps = Omit<OutputStoryTemplateProps, "output" | "status" | "children">;

export const outputStoryDecorators: Decorator[] = [
  (Story) => (
    <OutputStoryLayout>
      <Story />
    </OutputStoryLayout>
  ),
];

type GeneratedStory<TArgs extends object = Record<string, never>> = {
  args?: TArgs;
  name?: string;
  render: ((storyArgs: TArgs) => ReactNode) | (() => ReactNode);
};

export function createOutputStatusStory(
  output: TransitionMeta,
  status: TransitionRunStatus,
  shellProps: StoryShellProps = {},
): GeneratedStory {
  return {
    render: () => <OutputStoryTemplate output={output} status={status} {...shellProps} />,
  };
}

type CompletedOutputStoryConfig<TArgs extends object> = StoryShellProps & {
  output: TransitionMeta;
  args: TArgs;
  name?: string;
  renderContent: (args: TArgs) => ReactNode;
  renderShellProps?: (args: TArgs) => Partial<StoryShellProps>;
};

export function createCompletedOutputStory<TArgs extends object>({
  output,
  args,
  name,
  renderContent,
  renderShellProps,
  ...shellProps
}: CompletedOutputStoryConfig<TArgs>): GeneratedStory<TArgs> {
  return {
    ...(name ? { name } : {}),
    args,
    render: (storyArgs: TArgs) => {
      const dynamicShellProps = renderShellProps?.(storyArgs) ?? {};

      return (
        <OutputStoryTemplate
          output={output}
          status="completed"
          {...shellProps}
          {...dynamicShellProps}
        >
          {renderContent(storyArgs)}
        </OutputStoryTemplate>
      );
    },
  };
}
