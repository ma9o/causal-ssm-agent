import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { TooltipProvider } from "@/components/ui/tooltip";
import {
  ArtifactWorkspacePrototype,
  type ArtifactWorkspacePrototypeProps,
} from "./artifact-workspace-prototype";

const meta = {
  title: "Design Prototypes/Artifact Workspace",
  component: ArtifactWorkspacePrototype,
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "A disposable workspace shell for replacing the chronological pipeline feed with persistent Data and Model assets. The scientific surfaces reuse the production DAG, simulation, extraction, validation, and posterior-predictive components; their capabilities materialize as toggleable layers.",
      },
    },
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
  render: (args) => (
    <ArtifactWorkspacePrototype key={`${args.initialLens}-${args.materialization}`} {...args} />
  ),
  argTypes: {
    initialLens: {
      control: "inline-radio",
      options: ["data", "model", "split"],
    },
    materialization: {
      control: "inline-radio",
      options: ["structure", "measurement", "identified", "fitted", "simulated"],
    },
  },
} satisfies Meta<ArtifactWorkspacePrototypeProps>;

export default meta;

type Story = StoryObj<typeof meta>;

export const FullSimulation: Story = {
  name: "Full model + simulation",
  args: {
    initialLens: "model",
    materialization: "simulated",
  },
};

export const StructureOnly: Story = {
  name: "Progression 1 · structure only",
  args: {
    initialLens: "model",
    materialization: "structure",
  },
};

export const MeasurementMapped: Story = {
  name: "Progression 2 · measurement mapped",
  args: {
    initialLens: "model",
    materialization: "measurement",
  },
};

export const IdentifiedDesign: Story = {
  name: "Progression 3 · identified + marginalized",
  args: {
    initialLens: "model",
    materialization: "identified",
  },
};

export const FittedData: Story = {
  name: "Data asset + model fit",
  args: {
    initialLens: "data",
    materialization: "fitted",
  },
};

export const LinkedWorkspace: Story = {
  name: "Linked Data + Model",
  args: {
    initialLens: "split",
    materialization: "simulated",
  },
};
