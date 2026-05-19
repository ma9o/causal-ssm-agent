import type { Decorator } from "@storybook/nextjs-vite";
import { TooltipProvider } from "@/components/ui/tooltip";

export function withContainer(maxWidth = "max-w-4xl"): Decorator {
  function ContainerDecorator(Story: Parameters<Decorator>[0]) {
    return (
      <TooltipProvider>
        <div className={`${maxWidth} mx-auto p-4`}>
          <Story />
        </div>
      </TooltipProvider>
    );
  }

  return ContainerDecorator;
}
