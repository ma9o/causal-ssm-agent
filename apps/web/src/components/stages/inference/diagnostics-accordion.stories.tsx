import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5a, stage5b, stage5bNutsda } from "@/components/__fixtures__/inference-data";
import { DiagnosticsAccordion } from "./diagnostics-accordion";

const meta = {
  title: "Stages/Inference/DiagnosticsAccordion",
  component: DiagnosticsAccordion,
  decorators: [withContainer()],
} satisfies Meta<typeof DiagnosticsAccordion>;

export default meta;
type Story = StoryObj<typeof meta>;

export const SVIOnly: Story = {
  args: {
    sviDiagnostics: stage5a.svi_diagnostics,
    posteriorMarginals: stage5b.posterior_marginals,
    posteriorPairs: stage5b.posterior_pairs,
  },
};

export const MCMCOnly: Story = {
  args: {
    mcmcDiagnostics: stage5bNutsda.mcmc_diagnostics,
    posteriorMarginals: stage5bNutsda.posterior_marginals,
    posteriorPairs: stage5bNutsda.posterior_pairs,
  },
};

export const AllSections: Story = {
  args: {
    powerScaling: stage5bNutsda.power_scaling,
    ppc: stage5bNutsda.ppc,
    mcmcDiagnostics: stage5bNutsda.mcmc_diagnostics,
    looDiagnostics: stage5bNutsda.loo_diagnostics,
    posteriorMarginals: stage5bNutsda.posterior_marginals,
    posteriorPairs: stage5bNutsda.posterior_pairs,
  },
};

export const SMCWithLOO: Story = {
  args: {
    smcDiagnostics: stage5b.smc_diagnostics,
    looDiagnostics: stage5b.loo_diagnostics,
    powerScaling: stage5b.power_scaling,
    ppc: stage5b.ppc,
    posteriorMarginals: stage5b.posterior_marginals,
    posteriorPairs: stage5b.posterior_pairs,
  },
};

export const Empty: Story = {};
