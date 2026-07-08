import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import { withContainer } from "@/components/story-decorators";
import { posterior, posteriorAuxKalmanMCMC } from "@/components/__fixtures__/inference-data";
import { DiagnosticsAccordion } from "./diagnostics-accordion";

const meta = {
  title: "Pipeline/Outputs/Posterior/DiagnosticsAccordion",
  component: DiagnosticsAccordion,
  decorators: [withContainer()],
} satisfies Meta<typeof DiagnosticsAccordion>;

export default meta;
type Story = StoryObj<typeof meta>;

export const MCMCOnly: Story = {
  args: {
    mcmcDiagnostics: posteriorAuxKalmanMCMC.mcmc_diagnostics,
    posteriorMarginals: posteriorAuxKalmanMCMC.posterior_marginals,
    posteriorPairs: posteriorAuxKalmanMCMC.posterior_pairs,
  },
};

export const AllSections: Story = {
  args: {
    ppc: posteriorAuxKalmanMCMC.ppc,
    mcmcDiagnostics: posteriorAuxKalmanMCMC.mcmc_diagnostics,
    looDiagnostics: posteriorAuxKalmanMCMC.loo_diagnostics,
    posteriorMarginals: posteriorAuxKalmanMCMC.posterior_marginals,
    posteriorPairs: posteriorAuxKalmanMCMC.posterior_pairs,
  },
};

export const ParticleDiagnosticsWithLOO: Story = {
  args: {
    smcDiagnostics: posterior.smc_diagnostics,
    looDiagnostics: posterior.loo_diagnostics,
    ppc: posterior.ppc,
    posteriorMarginals: posterior.posterior_marginals,
    posteriorPairs: posterior.posterior_pairs,
  },
};

export const Empty: Story = {};
