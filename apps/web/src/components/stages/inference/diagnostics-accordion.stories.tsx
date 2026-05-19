import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { stage5b, stage5bAuxKalmanMCMC } from "@/components/__fixtures__/inference-data";
import { DiagnosticsAccordion } from "./diagnostics-accordion";

const meta = {
  title: "Stages/Inference/DiagnosticsAccordion",
  component: DiagnosticsAccordion,
  decorators: [withContainer()],
} satisfies Meta<typeof DiagnosticsAccordion>;

export default meta;
type Story = StoryObj<typeof meta>;

export const MCMCOnly: Story = {
  args: {
    mcmcDiagnostics: stage5bAuxKalmanMCMC.mcmc_diagnostics,
    posteriorMarginals: stage5bAuxKalmanMCMC.posterior_marginals,
    posteriorPairs: stage5bAuxKalmanMCMC.posterior_pairs,
  },
};

export const AllSections: Story = {
  args: {
    powerScaling: stage5bAuxKalmanMCMC.power_scaling,
    ppc: stage5bAuxKalmanMCMC.ppc,
    mcmcDiagnostics: stage5bAuxKalmanMCMC.mcmc_diagnostics,
    looDiagnostics: stage5bAuxKalmanMCMC.loo_diagnostics,
    posteriorMarginals: stage5bAuxKalmanMCMC.posterior_marginals,
    posteriorPairs: stage5bAuxKalmanMCMC.posterior_pairs,
  },
};

export const ParticleDiagnosticsWithLOO: Story = {
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
