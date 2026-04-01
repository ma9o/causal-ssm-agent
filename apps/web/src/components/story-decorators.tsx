import type { Meta } from "@storybook/nextjs-vite";
import { TooltipProvider } from "@/components/ui/tooltip";

type Decorator = NonNullable<Meta["decorators"]>[number];

export function withContainer(maxWidth = "max-w-4xl"): Decorator {
  return (Story) => (
    <TooltipProvider>
      <div className={`${maxWidth} mx-auto p-4`}>
        <Story />
      </div>
    </TooltipProvider>
  );
}
