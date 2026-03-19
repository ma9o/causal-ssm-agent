import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage5aData, Stage5bData } from "@causal-ssm/api-types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DiagnosticsAccordion } from "./diagnostics-accordion";
import sviFixture from "../../../../../../data/DOCTOLIB/run/stage-5b.json";
import nutsdaFixture from "../../../../../../data/DOCTOLIB/run/stage-5b-nutsda.json";
import stage5aFixture from "../../../../../../data/DOCTOLIB/run/stage-5a.json";

const svi = sviFixture as Stage5bData;
const nutsda = nutsdaFixture as Stage5bData;
const stage5a = stage5aFixture as Stage5aData;

const meta = {
  title: "Stages/Inference/DiagnosticsAccordion",
  component: DiagnosticsAccordion,
  decorators: [
    (Story) => (
      <TooltipProvider>
        <div className="max-w-4xl mx-auto p-4">
          <Story />
        </div>
      </TooltipProvider>
    ),
  ],
} satisfies Meta<typeof DiagnosticsAccordion>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SVIOnly: Story = {
  args: {
    sviDiagnostics: stage5a.svi_diagnostics,
    posteriorMarginals: svi.posterior_marginals,
    posteriorPairs: svi.posterior_pairs,
  },
};

export const MCMCOnly: Story = {
  args: {
    mcmcDiagnostics: nutsda.mcmc_diagnostics,
    posteriorMarginals: nutsda.posterior_marginals,
    posteriorPairs: nutsda.posterior_pairs,
  },
};

export const AllSections: Story = {
  args: {
    powerScaling: nutsda.power_scaling,
    ppc: nutsda.ppc,
    mcmcDiagnostics: nutsda.mcmc_diagnostics,
    looDiagnostics: nutsda.loo_diagnostics,
    posteriorMarginals: nutsda.posterior_marginals,
    posteriorPairs: nutsda.posterior_pairs,
  },
};

export const SMCWithLOO: Story = {
  args: {
    smcDiagnostics: svi.smc_diagnostics,
    looDiagnostics: svi.loo_diagnostics,
    powerScaling: svi.power_scaling,
    ppc: svi.ppc,
    posteriorMarginals: svi.posterior_marginals,
    posteriorPairs: svi.posterior_pairs,
  },
};

export const Empty: Story = {};
