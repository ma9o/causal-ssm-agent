import { FunctionalSpecLink } from "@/components/stages/model-spec/functional-spec-link";
import { ObsModelTable } from "@/components/stages/model-spec/obs-model-table";
import { Badge } from "@/components/ui/badge";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  confounderGroupLatex,
  confounderGroups,
  paramSymbol,
  priorLatex,
  stateEquationRows,
} from "@/lib/utils/ssm-latex";
import type {
  Indicator,
  LikelihoodSpec,
  ParameterSpec,
  PriorProposal,
} from "@nof1-causal-lab/api-types";
import katex from "katex";

interface SsmEquationDisplayProps {
  likelihoods: LikelihoodSpec[];
  parameters: ParameterSpec[];
  priors: PriorProposal[];
  indicators?: Indicator[];
}

/** Render a LaTeX string to an HTML string via KaTeX. */
function tex(latex: string, displayMode = true): string {
  return katex.renderToString(latex, {
    displayMode,
    throwOnError: false,
    strict: false,
  });
}

/** Render a confounder group's LaTeX to HTML via KaTeX. */
function confounderGroupHtml(group: Parameters<typeof confounderGroupLatex>[0]): string {
  return tex(confounderGroupLatex(group));
}

/** Inline KaTeX span. */
function Katex({ latex }: { latex: string }) {
  // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
  return <span dangerouslySetInnerHTML={{ __html: tex(latex, false) }} />;
}

function PriorSubtext({ parameterName, prior }: { parameterName: string; prior?: PriorProposal }) {
  if (!prior) {
    return (
      <div className="text-xs text-muted-foreground">
        <Katex latex={`${paramSymbol(parameterName)}:\\ \\text{Not authored}`} />
      </div>
    );
  }
  return (
    <div className="text-muted-foreground">
      <Katex latex={priorLatex(prior)} />
    </div>
  );
}

export function SSMEquationDisplay({
  likelihoods,
  parameters,
  priors,
  indicators,
}: SsmEquationDisplayProps) {
  const eqRows = stateEquationRows(parameters);
  const corrGroups = confounderGroups(parameters);
  const priorMap = new Map(priors.map((p) => [p.parameter, p]));

  // Generic form (kept as reference)
  const genericTransitionLatex = tex(
    String.raw`\begin{aligned}
\eta_i(t) &= \rho_i \, \eta_i(t\!-\!1) + \textstyle\sum_{j \in \mathrm{pa}(i)} \beta_{ji}\, \eta_j(t\!-\!1) + \varepsilon_i(t) \\
\varepsilon_i(t) &\sim \mathcal{N}(0,\, \sigma_i^2)
\end{aligned}`,
  );

  return (
    <div className="space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1">
          <h3 className="text-sm font-semibold">Stage 4 Semantic Equations</h3>
          <p className="max-w-3xl text-sm text-muted-foreground">
            This panel shows the interpretable Stage 4 discrete-time view used for prior
            elicitation. The compiler maps these semantic choices into the continuous-time runtime
            model used by the executable `SSMSpec`.
          </p>
        </div>
        <FunctionalSpecLink />
      </div>
      {/* ── State dynamics ── */}
      {eqRows.length > 0 && (
        <section>
          <div className="mb-2 flex items-center justify-between">
            <h4 className="inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              State Dynamics
              <StatTooltip explanation="Each latent state evolves as a discrete-time AR(1) process: it depends on its own previous value (persistence ρ), causal effects from parent states (β), and Gaussian noise." />
            </h4>
          </div>
          <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Semantic Stage 4 Form
            </p>
            {/* biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math */}
            <div dangerouslySetInnerHTML={{ __html: genericTransitionLatex }} />
          </div>
          <div className="mt-3 overflow-x-auto rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>State</TableHead>
                  <TableHead>
                    <span className="inline-flex items-center gap-1">
                      ρ (Persistence)
                      <StatTooltip explanation="Autoregressive coefficient controlling temporal persistence. Values near 1 mean high day-to-day persistence; near 0 means fast decay." />
                    </span>
                  </TableHead>
                  <TableHead>
                    <span className="inline-flex items-center gap-1">
                      β (Cross Effects)
                      <StatTooltip explanation="Directed causal effects from parent latent states, lagged by one time step." />
                    </span>
                  </TableHead>
                  <TableHead>
                    <span className="inline-flex items-center gap-1">
                      σ (Noise)
                      <StatTooltip explanation="Standard deviation of the innovation (process noise) driving this state's stochastic evolution." />
                    </span>
                  </TableHead>
                  <TableHead>
                    <span className="inline-flex items-center gap-1">
                      Initial State
                      <StatTooltip explanation="Distribution of the latent state at t = 0, and priors for its mean (μ₀) and standard deviation (σ₀)." />
                    </span>
                  </TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {eqRows.map((row) => {
                  const rhoPrior = priorMap.get(`rho_${row.state}`);
                  const sigmaPrior = priorMap.get(`sigma_${row.state}`);
                  const t0MeanPrior = priorMap.get(`t0_mean_${row.state}`);
                  const t0SdPrior = priorMap.get(`t0_sd_${row.state}`);

                  return (
                    <TableRow key={row.state}>
                      {/* State name */}
                      <TableCell className="whitespace-nowrap align-top">
                        <Katex latex={`\\eta_{\\text{${row.state.replace(/_/g, " ")}}}`} />
                      </TableCell>

                      {/* ρ (AR) — equation term + prior */}
                      <TableCell className="whitespace-nowrap align-top">
                        <div className="space-y-1">
                          <div>
                            <Katex latex={row.arTermLatex} />
                          </div>
                          <PriorSubtext parameterName={`rho_${row.state}`} prior={rhoPrior} />
                        </div>
                      </TableCell>

                      {/* β (cross effects) — each parent's term + prior */}
                      <TableCell className="whitespace-nowrap align-top">
                        {row.crossEffects.length > 0 ? (
                          <div className="space-y-2">
                            {row.crossEffects.map((ce) => {
                              const betaPrior = priorMap.get(`beta_${ce.source}_${row.state}`);
                              return (
                                <div key={ce.source} className="space-y-1">
                                  <div>
                                    <Katex latex={ce.termLatex} />
                                  </div>
                                  <PriorSubtext
                                    parameterName={`beta_${ce.source}_${row.state}`}
                                    prior={betaPrior}
                                  />
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </TableCell>

                      {/* σ (noise) — equation + prior */}
                      <TableCell className="whitespace-nowrap align-top">
                        <div className="space-y-1">
                          <div>
                            <Katex latex={row.noiseLatex} />
                          </div>
                          <PriorSubtext parameterName={`sigma_${row.state}`} prior={sigmaPrior} />
                        </div>
                      </TableCell>

                      {/* Initial state — equation + μ₀ & σ₀ priors */}
                      <TableCell className="whitespace-nowrap align-top">
                        <div className="space-y-1">
                          <div>
                            <Katex latex={row.initialLatex} />
                          </div>
                          <PriorSubtext
                            parameterName={`t0_mean_${row.state}`}
                            prior={t0MeanPrior}
                          />
                          <PriorSubtext parameterName={`t0_sd_${row.state}`} prior={t0SdPrior} />
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </section>
      )}

      {/* ── Correlated errors (per marginalized confounder) ── */}
      {corrGroups && (
        <section>
          <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Marginalized Confounders
            <StatTooltip explanation="Each unobserved confounder is marginalized out via nonparametric identification (front-door, IV, etc.). Its causal effect is absorbed into correlated residual noise among its observed children. Each block below shows one confounder and the joint noise structure it induces." />
          </h4>
          <div className="space-y-3">
            {corrGroups.map((group) => (
              <div
                key={group.confounder}
                className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3"
              >
                <div
                  // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
                  dangerouslySetInnerHTML={{
                    __html: confounderGroupHtml(group),
                  }}
                />
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ── Observation model ── */}
      {likelihoods.length > 0 && (
        <section>
          <h4 className="mb-2 inline-flex items-center gap-1.5 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Measurement Model
            <StatTooltip explanation="Maps latent states to observed indicators. Each variable has a distribution family (e.g. Gaussian, Poisson) and a link function (e.g. identity, log, logit) that transforms the linear predictor λᵀη(t) to the distribution's natural parameter." />
          </h4>
          <div className="overflow-x-auto rounded-md border bg-muted/30 px-4 py-3">
            <p className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              Semantic Stage 4 Form
            </p>
            <div
              // biome-ignore lint/security/noDangerouslySetInnerHtml: KaTeX renders sanitized math
              dangerouslySetInnerHTML={{
                __html: tex(
                  String.raw`\begin{aligned}
\mu_k(t) &= \boldsymbol{\lambda}_k^\top \boldsymbol{\eta}(t) \\[4pt]
\mathbb{E}[y_k(t)] &= g_k^{-1}\!\bigl(\mu_k(t)\bigr), \quad y_k(t) \sim \mathcal{F}_k
\end{aligned}`,
                ),
              }}
            />
          </div>
          <div className="mt-3">
            <ObsModelTable
              likelihoods={likelihoods}
              parameters={parameters}
              priors={priors}
              indicators={indicators}
            />
          </div>
        </section>
      )}
    </div>
  );
}
