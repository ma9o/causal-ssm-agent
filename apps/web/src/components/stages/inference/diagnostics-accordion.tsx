"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { formatNumber } from "@/lib/utils/format";
import type {
  LOODiagnostics,
  MCMCDiagnostics,
  PPCResult,
  PosteriorMarginal,
  PosteriorPair,
  PowerScalingResult,
  SVIDiagnostics,
} from "@causal-ssm/api-types";
import { ELBOLossChart } from "@/components/charts/elbo-loss-chart";
import { EnergyChart } from "@/components/charts/energy-chart";
import { LOOPITChart } from "@/components/charts/loo-pit-chart";
import { MCMCDiagnosticsPanel } from "@/components/charts/mcmc-diagnostics-panel";
import { ParetoKChart } from "@/components/charts/pareto-k-chart";
import { PosteriorDensityChart } from "@/components/charts/posterior-density-chart";
import { PosteriorPairsChart } from "@/components/charts/posterior-pairs-chart";
import { PowerScalingScatter } from "@/components/charts/power-scaling-scatter";
import { PowerScalingTable } from "./power-scaling-table";
import { PPCWarningsTable } from "./ppc-warnings-table";

interface DiagnosticsAccordionProps {
  powerScaling: PowerScalingResult[];
  ppc: PPCResult;
  mcmcDiagnostics?: MCMCDiagnostics | null;
  sviDiagnostics?: SVIDiagnostics | null;
  looDiagnostics?: LOODiagnostics | null;
  posteriorMarginals?: PosteriorMarginal[] | null;
  posteriorPairs?: PosteriorPair[] | null;
}

export function DiagnosticsAccordion({
  powerScaling,
  ppc,
  mcmcDiagnostics,
  sviDiagnostics,
  looDiagnostics,
  posteriorMarginals,
  posteriorPairs,
}: DiagnosticsAccordionProps) {
  const hasEnergy = mcmcDiagnostics?.energy != null;
  const hasMarginals = posteriorMarginals && posteriorMarginals.length > 0;
  const hasPairs = posteriorPairs && posteriorPairs.length > 0;

  const defaultOpen = ["mcmc", "svi", "ppc", "loo", "power-scaling"];

  return (
    <Accordion defaultOpen={defaultOpen}>
      {/* ── MCMC Diagnostics (convergence + energy + traces + rank histograms) ── */}
      {mcmcDiagnostics && (
        <AccordionItem value="mcmc">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              MCMC Diagnostics
              <StatTooltip explanation="Chain convergence (R-hat, ESS, MCSE), energy diagnostics, trace plots, and rank histograms for NUTS/HMC sampling." />
              <Badge variant={mcmcDiagnostics.num_divergences === 0 ? "success" : "destructive"}>
                {mcmcDiagnostics.num_divergences === 0
                  ? "Converged"
                  : `${mcmcDiagnostics.num_divergences} divergences`}
              </Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4">
              <MCMCDiagnosticsPanel diagnostics={mcmcDiagnostics} />
              {hasEnergy && <EnergyChart energy={mcmcDiagnostics.energy!} />}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── ELBO Convergence (SVI only) ── */}
      {sviDiagnostics && (
        <AccordionItem value="svi">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5">
              ELBO Convergence
              <StatTooltip explanation="Evidence Lower Bound loss over SVI optimization steps. Should decrease and plateau." />
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <ELBOLossChart diagnostics={sviDiagnostics} />
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── Posterior Predictive Checks (warnings + overlays + test stats) ── */}
      <AccordionItem value="ppc">
        <AccordionTrigger className="text-sm">
          <span className="inline-flex items-center gap-1.5 flex-wrap">
            Posterior Predictive Checks
            <StatTooltip explanation="Checks whether the fitted model can reproduce aspects of the observed data (distributional shape, variance, autocorrelation). Passing does not validate causal structure — only that the statistical model is not grossly misspecified." />
            <Badge
              variant={ppc.per_variable_warnings.every((w) => w.passed) ? "success" : "destructive"}
            >
              {ppc.per_variable_warnings.every((w) => w.passed) ? "Consistent" : "Misfit detected"}
            </Badge>
          </span>
        </AccordionTrigger>
        <AccordionContent>
          <div className="space-y-6">
            <PPCWarningsTable
              warnings={ppc.per_variable_warnings}
              testStats={ppc.test_stats ?? []}
              overlays={ppc.overlays ?? []}
            />
          </div>
        </AccordionContent>
      </AccordionItem>

      {/* ── LOO Cross-Validation (PIT + Pareto-K side by side) ── */}
      {looDiagnostics && (
        <AccordionItem value="loo">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              LOO Cross-Validation
              <StatTooltip explanation="LOO-CV via PSIS using one-step-ahead predictive log-likelihoods from the filter (innovation decomposition). Each 'observation' is one complete timestep, not individual cells. Valid for SSMs because the innovation sequence is conditionally independent given parameters." />
              <Badge
                variant={
                  looDiagnostics.n_bad_k != null && looDiagnostics.n_bad_k === 0
                    ? "success"
                    : "warning"
                }
              >
                ELPD = {formatNumber(looDiagnostics.elpd_loo, 1)}
              </Badge>
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-3">
              <div className="flex items-center gap-2 flex-wrap">
                <Badge variant="outline">ELPD = {formatNumber(looDiagnostics.elpd_loo, 1)}</Badge>
                <Badge variant="outline">p_loo = {formatNumber(looDiagnostics.p_loo, 1)}</Badge>
                <Badge variant="outline">SE = {formatNumber(looDiagnostics.se, 1)}</Badge>
                {looDiagnostics.n_bad_k != null && (
                  <Badge variant={looDiagnostics.n_bad_k === 0 ? "success" : "destructive"}>
                    {looDiagnostics.n_bad_k === 0
                      ? "All Pareto k OK"
                      : `${looDiagnostics.n_bad_k} bad Pareto k`}
                  </Badge>
                )}
              </div>
              <div className="grid gap-4 lg:grid-cols-2">
                {looDiagnostics.loo_pit && <LOOPITChart loo={looDiagnostics} />}
                {looDiagnostics.pareto_k && <ParetoKChart loo={looDiagnostics} />}
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      )}

      {/* ── Power Scaling (scatter + table side by side) ── */}
      <AccordionItem value="power-scaling">
        <AccordionTrigger className="text-sm">
          <span className="inline-flex items-center gap-1.5 flex-wrap">
            Power Scaling Diagnostics
            <StatTooltip explanation="Tests whether posteriors are driven by data (good) or priors (concerning). Scales the likelihood and prior to detect sensitivity." />
            {(() => {
              const nOk = powerScaling.filter((p) => p.diagnosis === "well_identified").length;
              const nPrior = powerScaling.filter((p) => p.diagnosis === "prior_dominated").length;
              const nConflict = powerScaling.filter(
                (p) => p.diagnosis === "prior_data_conflict",
              ).length;
              if (nOk === powerScaling.length) {
                return (
                  <Badge variant="success">
                    {nOk}/{powerScaling.length} OK
                  </Badge>
                );
              }
              return (
                <>
                  <Badge variant="success">
                    {nOk}/{powerScaling.length} OK
                  </Badge>
                  {nPrior > 0 && <Badge variant="warning">{nPrior} prior-dominated</Badge>}
                  {nConflict > 0 && (
                    <Badge variant="destructive">{nConflict} prior-data conflict</Badge>
                  )}
                </>
              );
            })()}
          </span>
        </AccordionTrigger>
        <AccordionContent>
          {powerScaling.length >= 2 ? (
            <div className="grid gap-4 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <PowerScalingTable results={powerScaling} />
              </div>
              <PowerScalingScatter results={powerScaling} />
            </div>
          ) : (
            <PowerScalingTable results={powerScaling} />
          )}
        </AccordionContent>
      </AccordionItem>

      {/* ── Posterior Exploration (marginals + pairs) ── */}
      {(hasMarginals || hasPairs) && (
        <AccordionItem value="posteriors">
          <AccordionTrigger className="text-sm">
            <span className="inline-flex items-center gap-1.5 flex-wrap">
              Posterior Exploration
              <StatTooltip explanation="Marginal posterior densities with 94% HDI, and pairwise scatter plots revealing parameter correlations and identifiability issues." />
            </span>
          </AccordionTrigger>
          <AccordionContent>
            <div className="space-y-4">
              {hasMarginals && (
                <div>
                  <h4 className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Marginal distributions
                  </h4>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {posteriorMarginals!.map((m) => (
                      <PosteriorDensityChart key={m.parameter} marginal={m} />
                    ))}
                  </div>
                </div>
              )}
              {hasPairs && (
                <div>
                  <h4 className="mb-2 text-xs font-medium text-muted-foreground uppercase tracking-wide">
                    Pairwise correlations
                  </h4>
                  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                    {posteriorPairs!.map((p) => (
                      <PosteriorPairsChart key={`${p.param_x}-${p.param_y}`} pair={p} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </AccordionContent>
        </AccordionItem>
      )}
    </Accordion>
  );
}
