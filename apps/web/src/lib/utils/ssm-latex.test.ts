import type { LikelihoodSpec, ParameterSpec, PriorProposal } from "@causal-ssm/api-types";
import { describe, expect, it } from "vitest";
import {
  concreteTransitionLines,
  confounderGroupLatex,
  confounderGroups,
  distName,
  extractConfounder,
  likelihoodLine,
  linkInverse,
  paramSymbol,
  parseCorrelation,
  parseFixedEffect,
  priorLine,
  stateEquationRows,
  stateNames,
  textify,
} from "./ssm-latex";
import type { ConfounderGroup } from "./ssm-latex";

describe("textify", () => {
  it("replaces underscores with spaces", () => {
    expect(textify("hello_world")).toBe("hello world");
  });

  it("handles no underscores", () => {
    expect(textify("hello")).toBe("hello");
  });

  it("handles multiple underscores", () => {
    expect(textify("a_b_c_d")).toBe("a b c d");
  });
});

describe("linkInverse", () => {
  it("identity returns predictor unchanged", () => {
    expect(linkInverse("identity", "\\mu")).toBe("\\mu");
  });

  it("log returns exp wrapper", () => {
    expect(linkInverse("log", "x")).toBe("\\exp(x)");
  });

  it("logit returns sigma wrapper", () => {
    expect(linkInverse("logit", "x")).toBe("\\sigma(x)");
  });

  it("probit returns Phi wrapper", () => {
    expect(linkInverse("probit", "x")).toBe("\\Phi(x)");
  });

  it("unknown link returns generic inverse", () => {
    expect(linkInverse("unknown", "x")).toBe("g^{-1}(x)");
  });
});

describe("distName", () => {
  it("maps gaussian to mathcal N", () => {
    expect(distName("gaussian")).toBe("\\mathcal{N}");
  });

  it("maps poisson", () => {
    expect(distName("poisson")).toBe("\\text{Poisson}");
  });

  it("maps bernoulli", () => {
    expect(distName("bernoulli")).toBe("\\text{Bernoulli}");
  });

  it("unknown dist uses text wrapper", () => {
    expect(distName("exotic")).toBe("\\text{exotic}");
  });
});

describe("likelihoodLine", () => {
  it("renders gaussian likelihood", () => {
    const lik = { variable: "mood", distribution: "gaussian", link: "identity" } as LikelihoodSpec;
    const result = likelihoodLine(lik);
    expect(result).toContain("\\mathcal{N}");
    expect(result).toContain("\\sigma");
  });

  it("renders poisson likelihood", () => {
    const lik = { variable: "steps", distribution: "poisson", link: "log" } as LikelihoodSpec;
    const result = likelihoodLine(lik);
    expect(result).toContain("\\text{Poisson}");
    expect(result).toContain("\\exp");
  });

  it("renders beta likelihood with phi", () => {
    const lik = { variable: "ratio", distribution: "beta", link: "logit" } as LikelihoodSpec;
    const result = likelihoodLine(lik);
    expect(result).toContain("\\text{Beta}");
    expect(result).toContain("\\phi");
  });

  it("includes construct name when provided", () => {
    const lik = { variable: "mood", distribution: "gaussian", link: "identity" } as LikelihoodSpec;
    const result = likelihoodLine(lik, "affect");
    expect(result).toContain("\\lambda");
    expect(result).toContain("affect");
  });
});

describe("paramSymbol", () => {
  it("converts beta prefix to greek", () => {
    expect(paramSymbol("beta_X_Y")).toContain("\\beta");
    expect(paramSymbol("beta_X_Y")).toContain("X Y");
  });

  it("converts sigma prefix to greek", () => {
    expect(paramSymbol("sigma_mood")).toContain("\\sigma");
    expect(paramSymbol("sigma_mood")).toContain("mood");
  });

  it("handles t0_mean prefix", () => {
    const result = paramSymbol("t0_mean_stress");
    expect(result).toContain("\\mu_{0");
    expect(result).toContain("stress");
  });

  it("handles t0_sd prefix", () => {
    const result = paramSymbol("t0_sd_sleep");
    expect(result).toContain("\\sigma_{0");
    expect(result).toContain("sleep");
  });

  it("unknown prefix uses text wrapper", () => {
    expect(paramSymbol("custom_param")).toContain("\\text{custom param}");
  });

  it("cor maps to psi", () => {
    expect(paramSymbol("cor_X_Y")).toContain("\\psi");
  });
});

describe("priorLine", () => {
  it("renders normal prior", () => {
    const prior = {
      parameter: "beta_X_Y",
      distribution: "Normal",
      params: { loc: 0, scale: 1 },
    } as PriorProposal;
    const result = priorLine(prior);
    expect(result).toContain("\\beta");
    expect(result).toContain("\\mathcal{N}");
    expect(result).toContain("0");
    expect(result).toContain("1");
  });

  it("renders half-normal prior", () => {
    const prior = {
      parameter: "sigma_mood",
      distribution: "HalfNormal",
      params: { scale: 2 },
    } as PriorProposal;
    const result = priorLine(prior);
    expect(result).toContain("\\text{HalfNormal}");
  });
});

describe("stateNames", () => {
  it("extracts from AR coefficients", () => {
    const params = [
      { name: "rho_stress", role: "ar_coefficient" },
      { name: "rho_sleep", role: "ar_coefficient" },
    ] as ParameterSpec[];
    expect(stateNames(params)).toEqual(["stress", "sleep"]);
  });

  it("falls back to residual_sd", () => {
    const params = [
      { name: "sigma_stress", role: "residual_sd" },
      { name: "sigma_sleep", role: "residual_sd" },
    ] as ParameterSpec[];
    expect(stateNames(params)).toEqual(["stress", "sleep"]);
  });

  it("returns empty for no matching params", () => {
    expect(stateNames([])).toEqual([]);
  });
});

describe("parseFixedEffect", () => {
  it("parses source and target", () => {
    const result = parseFixedEffect("beta_stress_sleep", ["stress", "sleep"]);
    expect(result).toEqual({ source: "stress", target: "sleep" });
  });

  it("returns null for no match", () => {
    expect(parseFixedEffect("beta_unknown", ["stress"])).toBeNull();
  });

  it("handles multi-word state names", () => {
    const result = parseFixedEffect("beta_daily_stress_sleep_quality", [
      "daily_stress",
      "sleep_quality",
    ]);
    expect(result).toEqual({ source: "daily_stress", target: "sleep_quality" });
  });
});

describe("parseCorrelation", () => {
  it("parses two states", () => {
    const result = parseCorrelation("cor_stress_sleep", ["stress", "sleep"]);
    expect(result).toEqual({ s1: "stress", s2: "sleep" });
  });

  it("returns null for unknown states", () => {
    expect(parseCorrelation("cor_a_b", ["stress", "sleep"])).toBeNull();
  });
});

describe("extractConfounder", () => {
  it("extracts confounder name from description", () => {
    const desc = "Correlation (marginalized confounder: genetics)";
    expect(extractConfounder(desc)).toBe("genetics");
  });

  it("returns null when no confounder", () => {
    expect(extractConfounder("Some other description")).toBeNull();
  });
});

describe("confounderGroups", () => {
  it("returns null for no correlation params", () => {
    const params = [{ name: "beta_X_Y", role: "fixed_effect" }] as ParameterSpec[];
    expect(confounderGroups(params)).toBeNull();
  });

  it("groups by confounder", () => {
    const params = [
      { name: "rho_stress", role: "ar_coefficient" },
      { name: "rho_sleep", role: "ar_coefficient" },
      {
        name: "cor_stress_sleep",
        role: "correlation",
        description: "Correlation (marginalized confounder: genetics)",
      },
    ] as ParameterSpec[];
    const groups = confounderGroups(params);
    expect(groups).not.toBeNull();
    expect(groups).toHaveLength(1);
    const first = groups?.[0];
    expect(first?.confounder).toBe("genetics");
    expect(first?.states).toContain("stress");
    expect(first?.states).toContain("sleep");
  });
});

describe("confounderGroupLatex", () => {
  it("renders aligned LaTeX block", () => {
    const group: ConfounderGroup = {
      confounder: "genetics",
      states: ["stress", "sleep"],
      pairs: [{ s1: "stress", s2: "sleep" }],
    };
    const result = confounderGroupLatex(group);
    expect(result).toContain("\\begin{aligned}");
    expect(result).toContain("\\end{aligned}");
    expect(result).toContain("genetics");
    expect(result).toContain("\\varepsilon");
    expect(result).toContain("\\psi");
  });
});

describe("stateEquationRows", () => {
  it("builds equation rows per state", () => {
    const params = [
      { name: "rho_stress", role: "ar_coefficient" },
      { name: "rho_sleep", role: "ar_coefficient" },
      { name: "beta_stress_sleep", role: "fixed_effect" },
      { name: "sigma_stress", role: "residual_sd" },
      { name: "sigma_sleep", role: "residual_sd" },
    ] as ParameterSpec[];

    const rows = stateEquationRows(params);
    expect(rows).toHaveLength(2);
    expect(rows[0].state).toBe("stress");
    expect(rows[1].state).toBe("sleep");
    expect(rows[1].crossEffects).toHaveLength(1);
    expect(rows[1].crossEffects[0].source).toBe("stress");
  });
});

describe("concreteTransitionLines", () => {
  it("builds transition lines", () => {
    const params = [
      { name: "rho_X", role: "ar_coefficient" },
      { name: "sigma_X", role: "residual_sd" },
    ] as ParameterSpec[];

    const lines = concreteTransitionLines(params);
    expect(lines.length).toBeGreaterThan(0);
    expect(lines.some((l) => l.includes("\\eta"))).toBe(true);
    expect(lines.some((l) => l.includes("\\varepsilon"))).toBe(true);
  });

  it("includes cross effects", () => {
    const params = [
      { name: "rho_X", role: "ar_coefficient" },
      { name: "rho_Y", role: "ar_coefficient" },
      { name: "beta_X_Y", role: "fixed_effect" },
      { name: "sigma_X", role: "residual_sd" },
      { name: "sigma_Y", role: "residual_sd" },
    ] as ParameterSpec[];

    const lines = concreteTransitionLines(params);
    expect(lines.some((l) => l.includes("\\beta"))).toBe(true);
  });
});
