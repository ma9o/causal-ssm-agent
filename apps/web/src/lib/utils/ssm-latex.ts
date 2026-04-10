import type {
  DistributionFamily,
  LikelihoodSpec,
  LinkFunction,
  ParameterSpec,
  PriorDistributionFamily,
  PriorProposal,
} from "@causal-ssm/api-types";

/** Convert snake_case to spaced text for use inside LaTeX \text{}. */
export function textify(name: string): string {
  return name.replace(/_/g, " ");
}

/** Build the g⁻¹(·) wrapper for a link function around a linear predictor. */
const LINK_INVERSE: Record<LinkFunction, (predictor: string) => string> = {
  identity: (p) => p,
  log: (p) => `\\exp(${p})`,
  inverse: (p) => `(${p})^{-1}`,
  logit: (p) => `\\sigma(${p})`,
  probit: (p) => `\\Phi(${p})`,
  cumulative_logit: (p) => `\\text{cumlogit}^{-1}(${p})`,
  softmax: (p) => `\\text{softmax}(${p})`,
};

export function linkInverse(link: string, predictor: string): string {
  const fn = LINK_INVERSE[link as LinkFunction];
  return fn ? fn(predictor) : `g^{-1}(${predictor})`;
}

/** Map distribution family enum to LaTeX name. */
const DIST_LATEX: Record<DistributionFamily, string> = {
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

export function distName(dist: string): string {
  return DIST_LATEX[dist as DistributionFamily] ?? `\\text{${dist}}`;
}

/** Build a single observation-model line, inlining the latent construct when known. */
export function likelihoodLine(
  lik: LikelihoodSpec,
  constructName?: string,
  options?: { includeMeasurementError?: boolean },
): string {
  const v = `\\text{${textify(lik.variable)}}`;
  const predictor = constructName
    ? `\\lambda_{${v}} \\, \\eta_{\\text{${textify(constructName)}}}(t)`
    : `\\mu_{${v}}`;
  const mu = linkInverse(lik.link, predictor);
  const d = distName(lik.distribution);
  const measurementErrorVariance = `\\sigma_{${v}}^{2}`;
  const includeMeasurementError = options?.includeMeasurementError ?? false;
  const withMeasurementError = (args: string) =>
    includeMeasurementError ? `${args},\\; ${measurementErrorVariance}` : args;

  if (lik.distribution === "gaussian" || lik.distribution === "student_t") {
    return `y_{${v}}(t) &\\sim ${d}(${mu},\\; ${measurementErrorVariance})`;
  }
  if (lik.distribution === "beta") {
    return `y_{${v}}(t) &\\sim ${d}(${withMeasurementError(
      `${mu}\\,\\phi,\\; (1 - ${mu})\\,\\phi`,
    )})`;
  }
  if (lik.distribution === "gamma") {
    return `y_{${v}}(t) &\\sim ${d}(${withMeasurementError(`\\kappa,\\; ${mu}/\\kappa`)})`;
  }
  if (lik.distribution === "negative_binomial") {
    return `y_{${v}}(t) &\\sim ${d}(${withMeasurementError(`r,\\; ${mu}`)})`;
  }
  return `y_{${v}}(t) &\\sim ${d}(${withMeasurementError(mu)})`;
}

/** Parse a parameter name into Greek letter + subscript. */
export function paramSymbol(name: string): string {
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
    tau: "\\tau",
    cor: "\\psi",
  };

  const parts = name.split("_");
  const greek = greekMap[parts[0]];
  if (greek && parts.length > 1) {
    return `${greek}_{\\text{${parts.slice(1).join(" ")}}}`;
  }
  return `\\text{${textify(name)}}`;
}

/** Strip the & alignment marker from a priorLine result for inline display. */
export function priorLatex(prior: PriorProposal): string {
  return priorLine(prior).replace(/&/g, "");
}

const PRIOR_DIST_LATEX: Record<PriorDistributionFamily, string> = {
  Normal: "\\mathcal{N}",
  HalfNormal: "\\text{HalfNormal}",
  Beta: "\\text{Beta}",
  Gamma: "\\text{Gamma}",
  Uniform: "\\text{Uniform}",
  TruncatedNormal: "\\text{TruncNormal}",
  Exponential: "\\text{Exp}",
  LogNormal: "\\text{LogNormal}",
};

function priorDistributionLatex(prior: PriorProposal): string {
  const vals = Object.values(prior.params).map((v) => String(v));
  const d =
    PRIOR_DIST_LATEX[prior.distribution as PriorDistributionFamily] ??
    `\\text{${prior.distribution}}`;
  return `${d}(${vals.join(",\\; ")})`;
}

/** Map a prior distribution name + params to LaTeX. */
export function priorLine(prior: PriorProposal): string {
  return `${paramSymbol(prior.parameter)} &\\sim ${priorDistributionLatex(prior)}`;
}

export function observationParameterSymbol({
  parameterName,
  likelihood,
}: {
  parameterName: string;
  likelihood: LikelihoodSpec;
}): string {
  const variableText = `\\text{${textify(likelihood.variable)}}`;

  if (parameterName.startsWith(`lambda_${likelihood.variable}_`)) {
    return `\\lambda_{${variableText}}`;
  }
  if (parameterName === `obs_sd_${likelihood.variable}`) {
    return `\\sigma_{${variableText}}`;
  }
  if (parameterName === "obs_df") {
    return "\\nu";
  }
  if (parameterName === "obs_shape") {
    return "\\kappa";
  }
  if (parameterName === "obs_r") {
    return "r";
  }
  if (parameterName === "obs_concentration") {
    return "\\phi";
  }
  if (parameterName === "obs_ordered_base") {
    return `\\boldsymbol{\\tau}_{${variableText}}`;
  }
  if (parameterName === "obs_ordered_gaps") {
    return `\\Delta\\boldsymbol{\\tau}_{${variableText}}`;
  }
  if (parameterName === "obs_cat_intercepts") {
    return `\\boldsymbol{\\alpha}_{${variableText}}`;
  }
  if (parameterName === "obs_cat_slopes") {
    return `\\boldsymbol{\\beta}_{${variableText}}`;
  }

  return paramSymbol(parameterName);
}

export function observationPriorLatex({
  prior,
  likelihood,
}: {
  prior: PriorProposal;
  likelihood: LikelihoodSpec;
}): string {
  return `${observationParameterSymbol({ parameterName: prior.parameter, likelihood })} \\sim ${priorDistributionLatex(prior)}`;
}

function observationParameterShownInLikelihood({
  parameterName,
  likelihood,
  hasConstruct,
}: {
  parameterName: string;
  likelihood: LikelihoodSpec;
  hasConstruct: boolean;
}): boolean {
  if (parameterName.startsWith(`lambda_${likelihood.variable}_`)) {
    return hasConstruct;
  }
  if (parameterName === `obs_sd_${likelihood.variable}`) {
    return likelihood.distribution === "gaussian" || likelihood.distribution === "student_t";
  }
  if (parameterName === "obs_df") {
    return likelihood.distribution === "student_t";
  }
  if (parameterName === "obs_shape") {
    return likelihood.distribution === "gamma";
  }
  if (parameterName === "obs_r") {
    return likelihood.distribution === "negative_binomial";
  }
  if (parameterName === "obs_concentration") {
    return likelihood.distribution === "beta";
  }
  return false;
}

export function observationParameterDefinitionLatex({
  parameterName,
  likelihood,
}: {
  parameterName: string;
  likelihood: LikelihoodSpec;
}): string {
  const symbol = observationParameterSymbol({ parameterName, likelihood });

  if (parameterName === `obs_sd_${likelihood.variable}`) {
    return `${symbol} &: \\text{measurement-error SD}`;
  }
  if (parameterName === "obs_df") {
    return `${symbol} &: \\text{Student-t degrees of freedom}`;
  }
  if (parameterName === "obs_shape") {
    return `${symbol} &: \\text{Gamma shape}`;
  }
  if (parameterName === "obs_r") {
    return `${symbol} &: \\text{negative-binomial dispersion}`;
  }
  if (parameterName === "obs_concentration") {
    return `${symbol} &: \\text{Beta concentration}`;
  }
  if (parameterName === "obs_ordered_base") {
    return `${symbol} &: \\text{ordered thresholds}`;
  }
  if (parameterName === "obs_ordered_gaps") {
    return `${symbol} &: \\text{threshold gaps}`;
  }
  if (parameterName === "obs_cat_intercepts") {
    return `${symbol} &: \\text{categorical intercepts}`;
  }
  if (parameterName === "obs_cat_slopes") {
    return `${symbol} &: \\text{categorical slopes}`;
  }

  return `${symbol}`;
}

export function observationEquationLatex({
  likelihood,
  constructName,
  parameterNames,
}: {
  likelihood: LikelihoodSpec;
  constructName?: string;
  parameterNames?: string[];
}): string {
  const measurementErrorParameterName = `obs_sd_${likelihood.variable}`;
  const hasMeasurementError = (parameterNames ?? []).includes(measurementErrorParameterName);
  const mainLine = likelihoodLine(likelihood, constructName, {
    includeMeasurementError:
      hasMeasurementError &&
      likelihood.distribution !== "gaussian" &&
      likelihood.distribution !== "student_t",
  });
  const supplementalLines = (parameterNames ?? [])
    .filter(
      (parameterName) =>
        !(parameterName === measurementErrorParameterName && hasMeasurementError) &&
        !observationParameterShownInLikelihood({
          parameterName,
          likelihood,
          hasConstruct: !!constructName,
        }),
    )
    .map((parameterName) =>
      observationParameterDefinitionLatex({
        parameterName,
        likelihood,
      }),
    );

  if (supplementalLines.length === 0) {
    return mainLine;
  }

  return `\\begin{aligned}${[mainLine, ...supplementalLines].join(" \\\\ ")}\\end{aligned}`;
}

/** Extract latent state names from AR coefficient parameters. */
export function stateNames(parameters: ParameterSpec[]): string[] {
  const ar = parameters.filter((p) => p.role === "ar_coefficient");
  if (ar.length > 0) {
    return ar.map((p) => p.name.split("_").slice(1).join("_"));
  }
  return parameters
    .filter((p) => p.role === "residual_sd")
    .map((p) => p.name.split("_").slice(1).join("_"));
}

/** Parse a fixed_effect parameter name into source→target given known state names. */
export function parseFixedEffect(
  name: string,
  knownStates: string[],
): { source: string; target: string } | null {
  const body = name.replace(/^beta_/, "");
  for (const state of [...knownStates].sort((a, b) => b.length - a.length)) {
    if (body.endsWith(`_${state}`)) {
      return { source: body.slice(0, -(state.length + 1)), target: state };
    }
  }
  return null;
}

/** Parse a cor_<s1>_<s2> parameter name into its two states. */
export function parseCorrelation(
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
export function extractConfounder(description: string): string | null {
  const m = description.match(/marginalized confounder:\s*(.+?)\)/);
  return m ? m[1] : null;
}

export interface ConfounderGroup {
  confounder: string;
  states: string[];
  pairs: { s1: string; s2: string }[];
}

/** Group correlation parameters by their source confounder. */
export function confounderGroups(parameters: ParameterSpec[]): ConfounderGroup[] | null {
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

/** Render LaTeX for a single confounder group (raw LaTeX, no KaTeX rendering). */
export function confounderGroupLatex(group: ConfounderGroup): string {
  const confTex = `\\text{${textify(group.confounder)}}`;
  const stateList = group.states.map((s) => `\\text{${textify(s)}}`).join(",\\, ");

  const lines: string[] = [];
  lines.push(`U_{${confTex}} &\\to \\{${stateList}\\}`);
  const epsilons = group.states.map((s) => `\\varepsilon_{\\text{${textify(s)}}}`).join(",\\, ");
  lines.push(`(${epsilons}) &\\sim \\mathcal{N}(\\mathbf{0},\\, \\Psi_{${confTex}})`);
  for (const { s1, s2 } of group.pairs) {
    const t1 = `\\text{${textify(s1)}}`;
    const t2 = `\\text{${textify(s2)}}`;
    lines.push(`\\psi_{${t1},\\,${t2}} &\\neq 0`);
  }

  return `\\begin{aligned}\n${lines.join(" \\\\\n")}\n\\end{aligned}`;
}

// ── Per-state equation fragments for table display ───────

export interface StateEquationRow {
  state: string;
  /** η(0) ~ N(μ₀, σ₀²) */
  initialLatex: string;
  /** ρ_s η_s(t-1) — the autoregressive term */
  arTermLatex: string;
  /** Each parent's cross-effect term */
  crossEffects: Array<{ source: string; termLatex: string }>;
  /** ε_s(t) ~ N(0, σ²) */
  noiseLatex: string;
}

/** Build per-state equation fragments for tabular display. */
export function stateEquationRows(parameters: ParameterSpec[]): StateEquationRow[] {
  const states = stateNames(parameters);
  const fixedEffects = parameters.filter((p) => p.role === "fixed_effect");

  const effectsByTarget = new Map<string, string[]>();
  for (const s of states) effectsByTarget.set(s, []);
  for (const fe of fixedEffects) {
    const parsed = parseFixedEffect(fe.name, states);
    if (parsed) effectsByTarget.get(parsed.target)?.push(parsed.source);
  }

  return states.map((state) => {
    const s = `\\text{${textify(state)}}`;
    const parents = effectsByTarget.get(state) ?? [];
    return {
      state,
      initialLatex: `\\eta_{${s}}(0) \\sim \\mathcal{N}(\\mu_{0,${s}},\\; \\sigma_{0,${s}}^{2})`,
      arTermLatex: `\\rho_{${s}} \\, \\eta_{${s}}(t\\!-\\!1)`,
      crossEffects: parents.map((src) => {
        const srcTex = `\\text{${textify(src)}}`;
        return {
          source: src,
          termLatex: `\\beta_{${srcTex} \\to ${s}} \\, \\eta_{${srcTex}}(t\\!-\\!1)`,
        };
      }),
      noiseLatex: `\\varepsilon_{${s}}(t) \\sim \\mathcal{N}(0,\\, \\sigma_{${s}}^2)`,
    };
  });
}

/** Build concrete per-state transition LaTeX lines from actual parameters. */
export function concreteTransitionLines(parameters: ParameterSpec[]): string[] {
  const states = stateNames(parameters);
  const fixedEffects = parameters.filter((p) => p.role === "fixed_effect");

  const effectsByTarget = new Map<string, string[]>();
  for (const s of states) effectsByTarget.set(s, []);

  for (const fe of fixedEffects) {
    const parsed = parseFixedEffect(fe.name, states);
    if (parsed) {
      effectsByTarget.get(parsed.target)?.push(parsed.source);
    }
  }

  const lines: string[] = [];

  for (const state of states) {
    const s = `\\text{${textify(state)}}`;
    lines.push(`\\eta_{${s}}(0) &\\sim \\mathcal{N}(\\mu_{0,${s}},\\; \\sigma_{0,${s}}^{2})`);
  }

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

  for (const state of states) {
    const s = `\\text{${textify(state)}}`;
    lines.push(`\\varepsilon_{${s}}(t) &\\sim \\mathcal{N}(0,\\, \\sigma_{${s}}^2)`);
  }

  return lines;
}
