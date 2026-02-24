import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatTooltip } from "@/components/ui/stat-tooltip";
import { FunctionalSpecLink } from "@/components/stages/model-spec/functional-spec-link";
import type { LikelihoodSpec, ParameterSpec, PriorProposal } from "@causal-ssm/api-types";
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

/** Convert snake_case to spaced text for use inside LaTeX \text{}. */
function textify(name: string): string {
  return name.replace(/_/g, " ");
}

/** Build the g⁻¹(·) wrapper for a link function around a linear predictor. */
function linkInverse(link: string, predictor: string): string {
  switch (link) {
    case "identity":
      return predictor;
    case "log":
      return `\\exp(${predictor})`;
    case "logit":
      return `\\sigma(${predictor})`;
    case "probit":
      return `\\Phi(${predictor})`;
    case "cumulative_logit":
      return `\\text{cumlogit}^{-1}(${predictor})`;
    case "softmax":
      return `\\text{softmax}(${predictor})`;
    default:
      return `g^{-1}(${predictor})`;
  }
}

/** Map distribution family enum to LaTeX name. */
function distName(dist: string): string {
  const map: Record<string, string> = {
    gaussian: "\\mathcal{N}",
    student_t: "t_{\\nu}",
    poisson: "\\text{Poisson}",
    gamma: "\\text{Gamma}",
    bernoulli: "\\text{Bernoulli}",
    negative_binomial: "\\text{NegBin}",
    beta: "\\text{Beta}",
    ordered_logistic: "\\text{OrdLogistic}",
    categorical: "\\text{Categorical}",
  };
  return map[dist] ?? `\\text{${dist}}`;
}

/** Build a single observation-model line with per-variable μ subscript. */
function likelihoodLine(lik: LikelihoodSpec): string {
  const v = `\\text{${textify(lik.variable)}}`;
  const mu = linkInverse(lik.link, `\\mu_{${v}}`);
  const d = distName(lik.distribution);

  if (lik.distribution === "gaussian" || lik.distribution === "student_t") {
    return `y_{${v}}(t) &\\sim ${d}(${mu},\\; \\sigma_{${v}}^{2})`;
  }
  if (lik.distribution === "beta") {
    return `y_{${v}}(t) &\\sim ${d}(${mu}\\,\\phi,\\; (1 - ${mu})\\,\\phi)`;
  }
  if (lik.distribution === "negative_binomial") {
    return `y_{${v}}(t) &\\sim ${d}(r,\\; ${mu})`;
  }
  // single-param: Poisson, Bernoulli, etc.
  return `y_{${v}}(t) &\\sim ${d}(${mu})`;
}

/** Parse a parameter name into Greek letter + subscript. */
function paramSymbol(name: string): string {
  // Initial state parameters: t0_mean_X → μ_{0,X}, t0_sd_X → σ_{0,X}
  if (name.startsWith("t0_mean_")) {
    const state = name.slice("t0_mean_".length);
    return `\\mu_{0,\\,\\text{${textify(state)}}}`;
  }
  if (name.startsWith("t0_sd_")) {
    const state = name.slice("t0_sd_".length);
    return `\\sigma_{0,\\,\\text{${textify(state)}}}`;
  }

  const greekMap: Record<string, string> = {
    beta: "\\beta",
    rho: "\\rho",
    sigma: "\\sigma",
    lambda: "\\lambda",
    alpha: "\\alpha",
    gamma: "\\gamma",
    phi: "\\phi",
    tau: "\\tau",
    mu: "\\mu",
    nu: "\\nu",
    kappa: "\\kappa",
    theta: "\\theta",
    omega: "\\omega",
    cor: "\\psi",
  };

  const parts = name.split("_");
  const greek = greekMap[parts[0]];
  if (greek && parts.length > 1) {
    return `${greek}_{\\text{${parts.slice(1).join(" ")}}}`;
  }
  return `\\text{${textify(name)}}`;
}

/** Map a prior distribution name + params to LaTeX. */
function priorLine(prior: PriorProposal): string {
  const sym = paramSymbol(prior.parameter);
  const vals = Object.values(prior.params).map((v) => String(v));

  const dMap: Record<string, string> = {
    Normal: "\\mathcal{N}",
    HalfNormal: "\\text{HalfNormal}",
    HalfCauchy: "\\text{HalfCauchy}",
    Beta: "\\text{Beta}",
    Gamma: "\\text{Gamma}",
    InverseGamma: "\\text{InvGamma}",
    Uniform: "\\text{Uniform}",
    Exponential: "\\text{Exp}",
    LKJCholesky: "\\text{LKJ}",
    Cauchy: "\\text{Cauchy}",
    LogNormal: "\\text{LogNormal}",
  };

  const d = dMap[prior.distribution] ?? `\\text{${prior.distribution}}`;
  return `${sym} &\\sim ${d}(${vals.join(",\\; ")})`;
}

/** Extract latent state names from AR coefficient parameters. */
function stateNames(parameters: ParameterSpec[]): string[] {
  const ar = parameters.filter((p) => p.role === "ar_coefficient");
  if (ar.length > 0) {
    return ar.map((p) => p.name.split("_").slice(1).join("_"));
  }
  // fallback: residual_sd params
  return parameters
    .filter((p) => p.role === "residual_sd")
    .map((p) => p.name.split("_").slice(1).join("_"));
}

/** Parse a fixed_effect parameter name into source→target given known state names. */
function parseFixedEffect(
  name: string,
  knownStates: string[],
): { source: string; target: string } | null {
  const body = name.replace(/^beta_/, "");
  // Match longest known state as suffix to handle multi-word state names
  for (const state of [...knownStates].sort((a, b) => b.length - a.length)) {
    if (body.endsWith(`_${state}`)) {
      return { source: body.slice(0, -(state.length + 1)), target: state };
    }
  }
  return null;
}

/** Parse a cor_<s1>_<s2> parameter name into its two states. */
function parseCorrelation(
  name: string,
  knownStates: string[],
): { s1: string; s2: string } | null {
  const body = name.replace(/^cor_/, "");
  for (const state1 of [...knownStates].sort((a, b) => b.length - a.length)) {
    if (body.startsWith(`${state1}_`)) {
      const rest = body.slice(state1.length + 1);
      if (knownStates.includes(rest)) {
        return { s1: state1, s2: rest };
      }
    }
  }
  return null;
}

/** Extract the marginalized confounder name from a correlation parameter description. */
function extractConfounder(description: string): string | null {
  const m = description.match(/marginalized confounder:\s*(.+?)\)/);
  return m ? m[1] : null;
}

interface ConfounderGroup {
  confounder: string;
  states: string[];
  pairs: { s1: string; s2: string }[];
}

/** Group correlation parameters by their source confounder. */
function confounderGroups(parameters: ParameterSpec[]): ConfounderGroup[] | null {
  const corParams = parameters.filter((p) => p.role === "correlation");
  if (corParams.length === 0) return null;

  const states = stateNames(parameters);
  const groups = new Map<string, { states: Set<string>; pairs: { s1: string; s2: string }[] }>();

  for (const p of corParams) {
    const parsed = parseCorrelation(p.name, states);
    if (!parsed) continue;
    const confounder = extractConfounder(p.description ?? "") ?? "unknown";
    let group = groups.get(confounder);
    if (!group) {
      group = { states: new Set(), pairs: [] };
      groups.set(confounder, group);
    }
    group.states.add(parsed.s1);
    group.states.add(parsed.s2);
    group.pairs.push(parsed);
  }

  if (groups.size === 0) return null;
  return [...groups.entries()].map(([confounder, { states: s, pairs }]) => ({
    confounder,
    states: [...s],
    pairs,
  }));
}

/** Render LaTeX for a single confounder group. */
function confounderGroupLatex(group: ConfounderGroup): string {
  const confTex = `\\text{${textify(group.confounder)}}`;
  const stateList = group.states.map((s) => `\\text{${textify(s)}}`).join(",\\, ");

  const lines: string[] = [];
  // Header: U_confounder → {children}
  lines.push(`U_{${confTex}} &\\to \\{${stateList}\\}`);
  // Joint noise statement
  const epsilons = group.states.map((s) => `\\varepsilon_{\\text{${textify(s)}}}`).join(",\\, ");
  lines.push(`(${epsilons}) &\\sim \\mathcal{N}(\\mathbf{0},\\, \\Psi_{${confTex}})`);
  // Non-zero off-diagonals
  for (const { s1, s2 } of group.pairs) {
    const t1 = `\\text{${textify(s1)}}`;
    const t2 = `\\text{${textify(s2)}}`;
    lines.push(`\\psi_{${t1},\\,${t2}} &\\neq 0`);
  }

  return tex(`\\begin{aligned}\n${lines.join(" \\\\\n")}\n\\end{aligned}`);
}

/** Build concrete per-state transition LaTeX lines from actual parameters. */
function concreteTransitionLines(parameters: ParameterSpec[]): string[] {
  const states = stateNames(parameters);
  const fixedEffects = parameters.filter((p) => p.role === "fixed_effect");

  // Group cross-lag effects by target state
  const effectsByTarget = new Map<string, string[]>();
  for (const s of states) effectsByTarget.set(s, []);

  for (const fe of fixedEffects) {
    const parsed = parseFixedEffect(fe.name, states);
    if (parsed) {
      effectsByTarget.get(parsed.target)?.push(parsed.source);
    }
  }

  const lines: string[] = [];

  // Initial state lines first
  for (const state of states) {
    const s = `\\text{${textify(state)}}`;
    lines.push(`\\eta_{${s}}(0) &\\sim \\mathcal{N}(\\mu_{0,${s}},\\; \\sigma_{0,${s}}^{2})`);
  }

  // Transition lines
  for (const state of states) {
    const s = `\\text{${textify(state)}}`;
    let rhs = `\\rho_{${s}} \\, \\eta_{${s}}(t\\!-\\!1)`;

    const parents = effectsByTarget.get(state) ?? [];
    for (const src of parents) {
      const srcTex = `\\text{${textify(src)}}`;
      rhs += ` + \\beta_{${srcTex} \\to ${s}} \\, \\eta_{${srcTex}}(t\\!-\\!1)`;
    }

    rhs += ` + \\varepsilon_{${s}}(t)`;
    lines.push(`\\eta_{${s}}(t) &= ${rhs}`);
  }

  // Add noise lines
  for (const state of states) {
    const s = `\\text{${textify(state)}}`;
    lines.push(`\\varepsilon_{${s}}(t) &\\sim \\mathcal{N}(0,\\, \\sigma_{${s}}^2)`);
  }

  return lines;
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
                  <div dangerouslySetInnerHTML={{ __html: confounderGroupLatex(group) }} />
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
