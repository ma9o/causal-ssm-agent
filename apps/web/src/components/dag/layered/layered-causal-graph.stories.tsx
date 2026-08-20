import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import {
  demoLatentStructure,
  demoMeasurementStructure,
  demoPosterior,
  demoStatisticalModelSpec,
} from "@/components/__fixtures__/demo-artifacts";
import { demoTraces } from "@/components/__fixtures__/demo-traces";
import {
  buildEdgePosteriors,
  buildPersistencePosteriors,
  buildBaselineReportScenarios,
} from "@/components/pipeline/output-views/baseline-report-scenarios";
import { TooltipProvider } from "@/components/ui/tooltip";
import { LayeredCausalGraph } from "./layered-causal-graph";
import type { LayeredCausalGraphModel } from "./layered-causal-graph-model";

const structureModel: LayeredCausalGraphModel = {
  structure: demoLatentStructure.latent_structure,
};

const measurementModel: LayeredCausalGraphModel = {
  ...structureModel,
  measurement: {
    measurement: demoMeasurementStructure.measurement_structure,
    knownInputs: demoMeasurementStructure.known_inputs,
    scientificOnlyConstructs: demoMeasurementStructure.scientific_only_constructs,
  },
};

const designModel: LayeredCausalGraphModel = {
  ...measurementModel,
  design: {
    causalDesign: demoMeasurementStructure.causal_design,
    structuralPlan: demoMeasurementStructure.structural_plan,
  },
};

const specificationModel: LayeredCausalGraphModel = {
  ...designModel,
  specification: {
    modelSpec: demoStatisticalModelSpec,
  },
};

const edgePosteriors = buildEdgePosteriors({
  latentStructure: demoLatentStructure,
  modelSpec: demoStatisticalModelSpec,
  posterior: demoPosterior,
});
const persistencePosteriors = buildPersistencePosteriors({
  modelSpec: demoStatisticalModelSpec,
  posterior: demoPosterior,
});

const fitModel: LayeredCausalGraphModel = {
  ...specificationModel,
  fit: {
    posterior: demoPosterior,
    edgePosteriors,
    persistencePosteriors,
  },
};

const scenarios = buildBaselineReportScenarios({ trace: demoTraces.baseline_report });
const simulationResult = scenarios[0]?.result;
if (!simulationResult) {
  throw new Error("The canonical DEMO trace must contain a materialized simulation result.");
}

const simulationModel: LayeredCausalGraphModel = {
  ...fitModel,
  simulation: { result: simulationResult },
};

const meta = {
  title: "DAG/Layered Causal Graph",
  component: LayeredCausalGraph,
  tags: ["svg-materialize"],
  parameters: {
    layout: "fullscreen",
    docs: {
      description: {
        component:
          "One stable structural graph with six explicit cumulative artifact layers. Every story uses the canonical DEMO fixture; later layers annotate the structural topology without replacing it.",
      },
    },
    svgMaterializer: {
      selector: 'svg[aria-label="Layered causal graph"]',
      auto: true,
    },
  },
  decorators: [
    (Story) => (
      <TooltipProvider>
        <Story />
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof LayeredCausalGraph>;

export default meta;

type Story = StoryObj<typeof meta>;

export const Structure: Story = {
  name: "1 · Structure",
  args: { model: structureModel },
};

export const Measurement: Story = {
  name: "2 · + Measurement",
  args: { model: measurementModel },
};

export const Design: Story = {
  name: "3 · + Design",
  args: { model: designModel },
};

export const Specification: Story = {
  name: "4 · + Specification",
  args: { model: specificationModel },
};

export const Fit: Story = {
  name: "5 · + Fit",
  args: { model: fitModel },
};

export const Simulation: Story = {
  name: "6 · + Simulation",
  args: { model: simulationModel },
};
