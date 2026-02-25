import type { LikelihoodSpec, ParameterSpec, PriorProposal } from "@causal-ssm/api-types";

/** Convert snake_case to spaced text for use inside LaTeX \text{}. */
export function textify(name: string): string {
  return name.replace(/_/g, " ");
}

/** Build the g⁻¹(·) wrapper for a link function around a linear predictor. */
export function linkInverse(link: string, predictor: string): string {
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
export function distName(dist: string): string {
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
export function likelihoodLine(lik: LikelihoodSpec): string {
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
  return `y_{${v}}(t) &\\sim ${d}(${mu})`;
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
export function priorLine(prior: PriorProposal): string {
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
