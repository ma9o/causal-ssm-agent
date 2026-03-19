import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Construct, Stage1aData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ConstructDetailPanel } from "./construct-detail-panel";
import fixture from "../../../../../../data/DOCTOLIB/run/stage-1a.json";

const data = fixture as unknown as Stage1aData;
const constructs = data.latent_model.constructs;
const endogenous = constructs.find((c) => c.role === "endogenous")!;
const exogenous = constructs.find((c) => c.role === "exogenous")!;
const outcome = constructs.find((c) => c.is_outcome)!;

const meta = {
  title: "Stages/LatentModel/ConstructDetailPanel",
  component: ConstructDetailPanel,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-md mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof ConstructDetailPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Endogenous: Story = {
  args: { construct: endogenous },
};

export const Exogenous: Story = {
  args: { construct: exogenous },
};

export const Outcome: Story = {
  args: { construct: outcome },
};
