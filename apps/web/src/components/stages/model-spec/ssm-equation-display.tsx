import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { FunctionalSpecLink } from "@/components/stages/model-spec/functional-spec-link";
import type { LikelihoodSpec, ParameterSpec, PriorProposal } from "@causal-ssm/api-types";
import {
  concreteTransitionLines,
  confounderGroupLatex,
  confounderGroups,
  likelihoodLine,
  priorLine,
} from "@/lib/utils/ssm-latex";
import katex from "katex";

interface SsmEquationDisplayProps {
  likelihoods: LikelihoodSpec[];
  parameters: ParameterSpec[];
  priors: PriorProposal[];
}

/** Render a LaTeX string to an HTML string via KaTeX. */
function tex(latex: string, displayMode = true): string {
  return katex.renderToString(latex, { displayMode, throwOnError: false, strict: false });
}

/** Render a confounder group's LaTeX to HTML via KaTeX. */
function confounderGroupHtml(group: Parameters<typeof confounderGroupLatex>[0]): string {
  return tex(confounderGroupLatex(group));
}

export function SSMEquationDisplay({ likelihoods, parameters, priors }: SsmEquationDisplayProps) {
  // --- State dynamics ---
  const transitionLines = concreteTransitionLines(parameters);
  const transitionLatex =
    transitionLines.length > 0
      ? tex(`\\begin{aligned}\n${transitionLines.join(" \\\\\n")}\n\\end{aligned}`)
      : null;

  // Generic form (kept as reference while iterating on the display)
  const genericTransitionLatex = tex(
    String.raw`\begin{aligned}
\eta_i(t) &= \rho_i \, \eta_i(t\!-\!1) + \textstyle\sum_{j \in \mathrm{pa}(i)} \beta_{ji}\, \eta_j(t\!-\!1) + \varepsilon_i(t) \\
\varepsilon_i(t) &\sim \mathcal{N}(0,\, \sigma_i^2)
\end{aligned}`,
  );

  // --- Correlated errors (from marginalized confounders) ---
  const corrGroups = confounderGroups(parameters);

  // --- Observation model ---
  const predictorDef =
    likelihoods.length > 0
      ? tex(
          String.raw`\mu_v(t) = \boldsymbol{\lambda}_v^\top \boldsymbol{\eta}(t)`,
        )
      : null;

  const obsLatex =
    likelihoods.length > 0
      ? tex(`\\begin{aligned}\n${likelihoods.map(likelihoodLine).join(" \\\\\n")}\n\\end{aligned}`)
      : null;

  // --- Priors ---
  const priorsLatex =
    priors.length > 0
      ? tex(`\\begin{aligned}\n${priors.map(priorLine).join(" \\\\\n")}\n\\end{aligned}`)
      : null;

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">SSM Equations</CardTitle>
          <FunctionalSpecLink />
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* State dynamics */}
        {transitionLatex && (
          <section>
            <div className="mb-2 flex items-center justify-between">
              <h4 className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                State Dynamics
                <StatTooltip explanation="Each latent state evolves as a discrete-time AR(1) process: it depends on its own previous value (persistence ρ), causal effects from parent states (β), and Gaussian noise." />
              </h4>
              <Badge variant="outline">Linear-Gaussian Dynamics</Badge>
            </div>
            <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
              <div dangerouslySetInnerHTML={{ __html: transitionLatex }} />
            </div>
            {/* TODO: remove generic reference once display is finalized */}
            <div className="mt-2 overflow-x-auto rounded-md border border-dashed bg-muted/15 px-4 py-3">
              <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Generic form (reference)</p>
              <div dangerouslySetInnerHTML={{ __html: genericTransitionLatex }} />
            </div>
          </section>
        )}

        {/* Correlated errors (per marginalized confounder) */}
        {corrGroups && (
          <section>
            <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Marginalized Confounders
              <StatTooltip explanation="Each unobserved confounder is marginalized out via nonparametric identification (front-door, IV, etc.). Its causal effect is absorbed into correlated residual noise among its observed children. Each block below shows one confounder and the joint noise structure it induces." />
            </h4>
            <div className="space-y-3">
              {corrGroups.map((group) => (
                <div key={group.confounder} className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
                  <div dangerouslySetInnerHTML={{ __html: confounderGroupHtml(group) }} />
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Observation model */}
        {obsLatex && (
          <section>
            <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Observation Model
              <StatTooltip explanation="Maps latent states to observed indicators. Each variable has a distribution family (e.g. Gaussian, Poisson) and a link function (e.g. identity, log, logit) that transforms the linear predictor λᵀη(t) to the distribution's natural parameter." />
            </h4>
            <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
              {predictorDef && <div dangerouslySetInnerHTML={{ __html: predictorDef }} />}
              <div dangerouslySetInnerHTML={{ __html: obsLatex }} />
            </div>
          </section>
        )}

        {/* Priors */}
        {priorsLatex && (
          <section>
            <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Priors
              <StatTooltip explanation="Informative or weakly informative prior distributions for each model parameter, elicited from domain literature. These constrain the posterior and encode existing knowledge about plausible effect sizes, persistence, and variance." />
            </h4>
            <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
              <div dangerouslySetInnerHTML={{ __html: priorsLatex }} />
            </div>
          </section>
        )}
      </CardContent>
    </Card>
  );
}
