import type {
  PriorProposal,
  Stage0Data,
  Stage1aData,
  Stage1bData,
  Stage2Data,
  Stage3Data,
  Stage4Data,
  Stage4bData,
  Stage5bData,
  Stage6Data,
} from "@causal-ssm/api-types";
import { STAGES } from "@causal-ssm/api-types";
import { CI_LOWER, CI_UPPER } from "@/lib/constants/diagnostics";
import { asciiDensity, asciiHistogram, asciiMultiLine, asciiScatter } from "./ascii-charts";
import { formatDateRange, formatNumber, formatPercent } from "./format";
import { quantile } from "./histogram";
import { mdTable } from "./markdown-tables";
import {
  concreteTransitionLines,
  confounderGroupLatex,
  confounderGroups,
  likelihoodLine,
  paramSymbol,
  priorLine,
  stateNames,
  textify,
} from "./ssm-latex";

export interface AllStageData {
  "stage-0"?: Stage0Data | null;
  "stage-1a"?: Stage1aData | null;
  "stage-1b"?: Stage1bData | null;
  "stage-2"?: Stage2Data | null;
  "stage-3"?: Stage3Data | null;
  "stage-4"?: Stage4Data | null;
  "stage-4b"?: Stage4bData | null;
  "stage-5b"?: Stage5bData | null;
  "stage-6"?: Stage6Data | null;
}

function section(level: number, title: string): string {
  return `${"#".repeat(level)} ${title}`;
}

function fenced(content: string, lang = ""): string {
  return `\`\`\`${lang}\n${content}\n\`\`\``;
}

function latex(content: string): string {
  return `$$\n${content}\n$$`;
}

export function generateMarkdown(data: AllStageData, userId: string): string {
  const lines: string[] = [];

  // --- Header ---
  lines.push("# Causal Inference Pipeline Report");
  lines.push("");
  lines.push(`**User ID**: \`${userId}\``);
  lines.push(`**Generated**: ${new Date().toISOString().slice(0, 10)}`);
  lines.push("");
  lines.push("---");
  lines.push("");

  // --- Stage 0: Preprocess ---
  const s0 = data["stage-0"];
  if (s0) {
    lines.push(section(2, `Stage 0: ${STAGES[0].label}`));
    lines.push(`> ${STAGES[0].description}`);
    lines.push("");
    lines.push(`- **Records**: ${s0.n_records.toLocaleString()}`);
    lines.push(`- **Date range**: ${formatDateRange(s0.date_range.start, s0.date_range.end)}`);
    lines.push("");

    if (s0.sample.length > 0) {
      lines.push(section(3, "Data Sample (first 10 rows)"));
      lines.push("");
      const cols = Object.keys(s0.sample[0]);
      const rows = s0.sample.slice(0, 10).map((row) =>
        cols.map((c) => row[c] ?? ""),
      );
      lines.push(mdTable(cols, rows));
      lines.push("");
    }
    lines.push("---");
    lines.push("");
  }

  // --- Stage 1a: Latent Model ---
  const s1a = data["stage-1a"];
  if (s1a) {
    lines.push(section(2, `Stage 1a: ${STAGES[1].label}`));
    lines.push(`> ${STAGES[1].description}`);
    lines.push("");
    lines.push(`- **Outcome**: ${s1a.outcome_name}`);
    lines.push(`- **Treatments**: ${s1a.treatments.join(", ")}`);
    lines.push("");

    lines.push(section(3, "Constructs"));
    lines.push("");
    const constructRows = s1a.latent_model.constructs.map((c) => [
      c.name,
      c.description,
      c.role,
      c.temporal_status,
      c.is_outcome ? "Yes" : "No",
    ]);
    lines.push(mdTable(["Name", "Description", "Role", "Temporal", "Outcome"], constructRows));
    lines.push("");

    lines.push(section(3, "Causal Edges"));
    lines.push("");
    const edgeRows = s1a.latent_model.edges.map((e) => [
      e.cause,
      e.effect,
      e.lagged ? "Yes" : "No",
      e.description,
    ]);
    lines.push(mdTable(["Cause", "Effect", "Lagged", "Description"], edgeRows));
    lines.push("");
    lines.push("---");
    lines.push("");
  }

  // --- Stage 1b: Measurement & Nonparametric ID ---
  const s1b = data["stage-1b"];
  if (s1b) {
    lines.push(section(2, `Stage 1b: ${STAGES[2].label}`));
    lines.push(`> ${STAGES[2].description}`);
    lines.push("");

    const spec = s1b.causal_spec;

    // Gate alert
    if (s1b.outcome === "fail") {
      lines.push("> **GATE BLOCKED**: Non-identifiable treatment effects detected.");
      lines.push("");
    }

    // Non-identifiable treatments
    const nonId = spec.identifiability?.non_identifiable_treatments ?? {};
    const nonIdEntries = Object.entries(nonId);
    if (nonIdEntries.length > 0) {
      lines.push(section(3, "Non-Identifiable Treatments"));
      lines.push("");
      for (const [name, status] of nonIdEntries) {
        lines.push(`- **${name}**: confounded by ${status?.confounders.join(", ") ?? "unknown"}`);
      }
      lines.push("");
    }

    // Identifiable treatments
    const idTx = spec.identifiability?.identifiable_treatments ?? {};
    const idEntries = Object.entries(idTx);
    if (idEntries.length > 0) {
      lines.push(section(3, "Identifiable Treatments"));
      lines.push("");
      const idRows = idEntries.map(([name, status]) => [
        name,
        status?.method ?? "—",
        status?.estimand ?? "—",
      ]);
      lines.push(mdTable(["Treatment", "Method", "Estimand"], idRows));
      lines.push("");
    }

    // Indicators
    lines.push(section(3, "Indicators"));
    lines.push("");
    const indRows = spec.measurement.indicators.map((ind) => [
      ind.name,
      ind.construct_name,
      ind.measurement_dtype,
      ind.aggregation,
      ind.how_to_measure,
    ]);
    lines.push(mdTable(["Indicator", "Construct", "Type", "Aggregation", "How to Measure"], indRows));
    lines.push("");
    lines.push("---");
    lines.push("");
  }

  // --- Stage 2: Data Extraction ---
  const s2 = data["stage-2"];
  if (s2) {
    lines.push(section(2, `Stage 2: ${STAGES[3].label}`));
    lines.push(`> ${STAGES[3].description}`);
    lines.push("");

    const succeeded = s2.workers.filter((w) => w.status === "completed").length;
    const failed = s2.workers.filter((w) => w.status === "failed").length;
    const total = s2.workers.length;

    lines.push(`- **Workers**: ${succeeded} succeeded, ${failed} failed, ${total} total`);
    lines.push("");

    // Per-indicator counts
    const countEntries = Object.entries(s2.per_indicator_counts);
    if (countEntries.length > 0) {
      lines.push(section(3, "Extractions per Indicator"));
      lines.push("");
      const countRows = countEntries.map(([name, count]) => [name, String(count ?? 0)]);
      lines.push(mdTable(["Indicator", "Count"], countRows));
      lines.push("");
    }

    // Failed worker errors
    const errors = s2.workers.filter((w) => w.status === "failed" && w.error);
    if (errors.length > 0) {
      lines.push(section(3, "Errors"));
      lines.push("");
      for (const w of errors) {
        lines.push(`- Worker ${w.worker_id}: ${w.error}`);
      }
      lines.push("");
    }
    lines.push("---");
    lines.push("");
  }

  // --- Stage 3: Validation ---
  const s3 = data["stage-3"];
  if (s3) {
    lines.push(section(2, `Stage 3: ${STAGES[4].label}`));
    lines.push(`> ${STAGES[4].description}`);
    lines.push("");

    const report = s3.validation_report;

    if (!report.is_valid) {
      lines.push("> **GATE BLOCKED**: Data validation failed.");
      lines.push("");
    }

    if (report.per_indicator_health.length > 0) {
      lines.push(section(3, "Indicator Health"));
      lines.push("");
      const healthRows = report.per_indicator_health.map((h) => [
        h.indicator,
        String(h.n_obs),
        h.variance != null ? formatNumber(h.variance) : "—",
        h.time_coverage_ratio != null ? formatPercent(h.time_coverage_ratio) : "—",
        h.max_gap_ratio != null ? formatPercent(h.max_gap_ratio) : "—",
        String(h.dtype_violations),
        formatPercent(h.duplicate_pct),
      ]);
      lines.push(
        mdTable(
          ["Indicator", "N obs", "Variance", "Coverage", "Max Gap", "Type Violations", "Duplicates"],
          healthRows,
        ),
      );
      lines.push("");
    }
    lines.push("---");
    lines.push("");
  }

  // --- Stage 4: Model Specification ---
  const s4 = data["stage-4"];
  if (s4) {
    lines.push(section(2, `Stage 4: ${STAGES[5].label}`));
    lines.push(`> ${STAGES[5].description}`);
    lines.push("");

    // SSM equations as LaTeX
    const transitionLns = concreteTransitionLines(s4.model_spec.parameters);
    if (transitionLns.length > 0) {
      lines.push(section(3, "State Dynamics"));
      lines.push("");
      lines.push(latex(`\\begin{aligned}\n${transitionLns.join(" \\\\\n")}\n\\end{aligned}`));
      lines.push("");
    }

    // Marginalized confounders
    const corrGroups = confounderGroups(s4.model_spec.parameters);
    if (corrGroups) {
      lines.push(section(3, "Marginalized Confounders"));
      lines.push("");
      for (const group of corrGroups) {
        lines.push(latex(confounderGroupLatex(group)));
        lines.push("");
      }
    }

    // Observation model
    const s1b = data["stage-1b"];
    const indMap = s1b
      ? Object.fromEntries(s1b.causal_spec.measurement.indicators.map((ind) => [ind.name, ind.construct_name]))
      : undefined;
    if (s4.model_spec.likelihoods.length > 0) {
      lines.push(section(3, "Observation Model"));
      lines.push("");
      if (!indMap) {
        lines.push(latex("\\mu_v(t) = \\boldsymbol{\\lambda}_v^\\top \\boldsymbol{\\eta}(t)"));
        lines.push("");
      }
      lines.push(
        latex(
          `\\begin{aligned}\n${s4.model_spec.likelihoods.map((l) => likelihoodLine(l, indMap?.[l.variable])).join(" \\\\\n")}\n\\end{aligned}`,
        ),
      );
      lines.push("");

      // Measurement table
      lines.push(section(3, "Measurement Model"));
      lines.push("");
      const measRows = s4.model_spec.likelihoods.map((l) => [
        l.variable,
        l.distribution,
        l.link,
        l.reasoning,
      ]);
      lines.push(mdTable(["Variable", "Distribution", "Link", "Reasoning"], measRows));
      lines.push("");
    }

    // Build initial state priors
    const states = stateNames(s4.model_spec.parameters);
    const t0Priors: PriorProposal[] = [];
    for (const s of states) {
      t0Priors.push({
        parameter: `t0_mean_${s}`,
        distribution: "Normal",
        params: { mu: 0, sigma: 2 },
        sources: [],
        reasoning: `Default weakly informative prior for the initial state mean of ${textify(s)}.`,
      });
      t0Priors.push({
        parameter: `t0_sd_${s}`,
        distribution: "HalfNormal",
        params: { sigma: 2 },
        sources: [],
        reasoning: `Default weakly informative prior for the initial state standard deviation of ${textify(s)}.`,
      });
    }
    const allPriors: PriorProposal[] = [
      ...Object.values(s4.priors).filter((p): p is PriorProposal => p != null),
      ...t0Priors,
    ];

    // Priors as LaTeX
    if (allPriors.length > 0) {
      lines.push(section(3, "Prior Distributions"));
      lines.push("");
      lines.push(
        latex(`\\begin{aligned}\n${allPriors.map(priorLine).join(" \\\\\n")}\n\\end{aligned}`),
      );
      lines.push("");

      // Priors table with sources
      const priorRows = allPriors.map((p) => {
        const params = Object.entries(p.params)
          .map(([k, v]) => `${k}=${v}`)
          .join(", ");
        const sourceLinks = p.sources
          .map((s) => (s.url ? `[${s.title}](${s.url})` : s.title))
          .join("; ");
        return [
          `$${paramSymbol(p.parameter)}$`,
          `${p.distribution}(${params})`,
          sourceLinks || "—",
          p.reasoning,
        ];
      });
      lines.push(mdTable(["Parameter", "Prior", "Sources", "Reasoning"], priorRows));
      lines.push("");
    }
    lines.push("---");
    lines.push("");
  }

  // --- Stage 4b: Parametric Identifiability ---
  const s4b = data["stage-4b"];
  if (s4b) {
    lines.push(section(2, `Stage 4b: ${STAGES[6].label}`));
    lines.push(`> ${STAGES[6].description}`);
    lines.push("");

    const pid = s4b.parametric_id;

    if (pid.t_rule && !pid.t_rule.satisfies) {
      lines.push(
        `> **GATE BLOCKED**: T-Rule violated — ${pid.t_rule.n_free_params} free parameters but only ${pid.t_rule.n_moments} moment conditions.`,
      );
      lines.push("");
    }

    if (pid.t_rule) {
      lines.push(section(3, "T-Rule"));
      lines.push("");
      lines.push(`- **Free parameters**: ${pid.t_rule.n_free_params}`);
      if (pid.t_rule.n_manifest != null) {
        lines.push(`- **Manifest variables**: ${pid.t_rule.n_manifest}`);
      }
      if (pid.t_rule.n_timepoints != null) {
        lines.push(`- **Timepoints**: ${pid.t_rule.n_timepoints}`);
      }
      lines.push(`- **Moment conditions**: ${pid.t_rule.n_moments}`);
      lines.push(`- **Satisfies**: ${pid.t_rule.satisfies ? "Yes" : "No"}`);
      lines.push("");
    }

    // RB partition
    if (s4b.rb_partition) {
      lines.push(section(3, "Rao-Blackwellization Partition"));
      lines.push("");
      const kalman = s4b.rb_partition.latent_variables
        .filter((v) => v.method === "kalman")
        .map((v) => v.name);
      const particle = s4b.rb_partition.latent_variables
        .filter((v) => v.method === "particle")
        .map((v) => v.name);
      if (kalman.length > 0) lines.push(`- **Kalman**: ${kalman.join(", ")}`);
      if (particle.length > 0) lines.push(`- **Particle**: ${particle.join(", ")}`);
      lines.push("");
    }

    // Per-parameter classification
    if (pid.per_param_classification && pid.per_param_classification.length > 0) {
      lines.push(section(3, "Parameter Classification"));
      lines.push("");
      const classRows = pid.per_param_classification.map((p) => [
        p.name,
        p.classification,
        p.contraction_ratio != null ? formatNumber(p.contraction_ratio) : "—",
      ]);
      lines.push(mdTable(["Parameter", "Classification", "Contraction Ratio"], classRows));
      lines.push("");
    }
    lines.push("---");
    lines.push("");
  }

  // --- Stage 5: Inference & Diagnostics ---
  const s5 = data["stage-5b"];
  if (s5) {
    lines.push(section(2, `Stage 5b: ${STAGES[8].label}`));
    lines.push(`> ${STAGES[8].description}`);
    lines.push("");

    // MCMC diagnostics
    if (s5.mcmc_diagnostics) {
      const mcmc = s5.mcmc_diagnostics;
      lines.push(section(3, "MCMC Diagnostics"));
      lines.push("");
      lines.push(`- **Divergences**: ${mcmc.num_divergences} (${formatPercent(mcmc.divergence_rate)})`);
      lines.push(`- **Tree depth**: mean=${formatNumber(mcmc.tree_depth_mean, 1)}, max=${mcmc.tree_depth_max}`);
      lines.push(`- **Accept prob**: ${formatNumber(mcmc.accept_prob_mean)}`);
      if (mcmc.num_chains) lines.push(`- **Chains**: ${mcmc.num_chains}`);
      if (mcmc.num_samples) lines.push(`- **Samples**: ${mcmc.num_samples}`);
      lines.push("");

      // Per-parameter convergence table
      if (mcmc.per_parameter.length > 0) {
        lines.push(section(4, "Convergence"));
        lines.push("");
        const convRows = mcmc.per_parameter.map((p) => {
          const rhat = Array.isArray(p.r_hat) ? p.r_hat.map((v) => formatNumber(v)).join(", ") : formatNumber(p.r_hat);
          const ess = Array.isArray(p.ess_bulk) ? p.ess_bulk.map((v) => formatNumber(v, 0)).join(", ") : formatNumber(p.ess_bulk, 0);
          const essTail = p.ess_tail
            ? (Array.isArray(p.ess_tail) ? p.ess_tail.map((v) => formatNumber(v, 0)).join(", ") : formatNumber(p.ess_tail, 0))
            : "—";
          return [p.parameter, rhat, ess, essTail];
        });
        lines.push(mdTable(["Parameter", "R-hat", "ESS bulk", "ESS tail"], convRows));
        lines.push("");
      }

      // Trace plots (ASCII)
      if (mcmc.trace_data && mcmc.trace_data.length > 0) {
        lines.push(section(4, "Trace Plots"));
        lines.push("");
        for (const trace of mcmc.trace_data) {
          const series = trace.chains.map((c) => c.values);
          lines.push(fenced(asciiMultiLine(series, { label: trace.parameter, height: 10, width: 60 })));
          lines.push("");
        }
      }
    }

    // SVI diagnostics
    if (s5.svi_diagnostics) {
      lines.push(section(3, "SVI / ELBO Convergence"));
      lines.push("");
      lines.push(
        fenced(
          asciiDensity(
            s5.svi_diagnostics.elbo_losses.map((_, i) => i),
            s5.svi_diagnostics.elbo_losses,
            { label: "ELBO loss over optimization steps", height: 10 },
          ),
        ),
      );
      lines.push("");
    }

    // PPC
    lines.push(section(3, "Posterior Predictive Checks"));
    lines.push("");
    const allPassed = s5.ppc.per_variable_warnings.every((w) => w.passed);
    lines.push(`**Status**: ${allPassed ? "Consistent" : "Misfit detected"}`);
    lines.push("");

    if (s5.ppc.per_variable_warnings.length > 0) {
      const ppcRows = s5.ppc.per_variable_warnings.map((w) => [
        w.variable,
        w.check_type,
        w.passed ? "Pass" : "Fail",
        formatNumber(w.value),
        w.message,
      ]);
      lines.push(mdTable(["Variable", "Check", "Result", "Value", "Message"], ppcRows));
      lines.push("");
    }

    // LOO diagnostics
    if (s5.loo_diagnostics) {
      const loo = s5.loo_diagnostics;
      lines.push(section(3, "LOO Cross-Validation"));
      lines.push("");
      lines.push(`- **ELPD**: ${formatNumber(loo.elpd_loo, 1)}`);
      lines.push(`- **p_loo**: ${formatNumber(loo.p_loo, 1)}`);
      lines.push(`- **SE**: ${formatNumber(loo.se, 1)}`);
      lines.push(`- **Data points**: ${loo.n_data_points}`);
      if (loo.n_bad_k != null) {
        lines.push(`- **Bad Pareto k**: ${loo.n_bad_k}`);
      }
      lines.push("");
    }

    // Power scaling
    if (s5.power_scaling.length > 0) {
      lines.push(section(3, "Power Scaling Diagnostics"));
      lines.push("");
      const psRows = s5.power_scaling.map((p) => [
        p.parameter,
        p.diagnosis,
        formatNumber(p.prior_sensitivity),
        formatNumber(p.likelihood_sensitivity),
        p.psis_k_hat != null ? formatNumber(p.psis_k_hat) : "—",
      ]);
      lines.push(mdTable(["Parameter", "Diagnosis", "Prior Sens.", "Likelihood Sens.", "PSIS k-hat"], psRows));
      lines.push("");

      // ASCII scatter
      if (s5.power_scaling.length >= 2) {
        const scatterPoints = s5.power_scaling.map((p) => ({
          x: p.prior_sensitivity,
          y: p.likelihood_sensitivity,
          label: p.parameter,
        }));
        lines.push(fenced(asciiScatter(scatterPoints, { label: "Power Scaling (prior vs likelihood sensitivity)" })));
        lines.push("");
      }
    }

    // Posterior marginals (ASCII density)
    if (s5.posterior_marginals && s5.posterior_marginals.length > 0) {
      lines.push(section(3, "Posterior Marginals"));
      lines.push("");
      for (const m of s5.posterior_marginals) {
        lines.push(
          fenced(
            asciiDensity(m.x_values, m.density, {
              label: `${m.parameter}  (mean=${formatNumber(m.mean)}, sd=${formatNumber(m.sd)}, HDI=[${formatNumber(m.hdi_3)}, ${formatNumber(m.hdi_97)}])`,
              height: 10,
              width: 60,
            }),
          ),
        );
        lines.push("");
      }
    }

    // Inference metadata
    lines.push(
      `*Inference: ${s5.inference_metadata.method} | ${s5.inference_metadata.n_samples} samples | ${s5.inference_metadata.duration_seconds.toFixed(1)}s*`,
    );
    lines.push("");
    lines.push("---");
    lines.push("");
  }

  // --- Stage 6: Treatment Effects ---
  const s6 = data["stage-6"];
  if (s6) {
    lines.push(section(2, `Stage 6: ${STAGES[9].label}`));
    lines.push(`> ${STAGES[9].description}`);
    lines.push("");

    if (s6.intervention_results.length === 0) {
      lines.push("No treatment effects were estimated.");
      lines.push("");
    } else {
      // Sort by |effect_size| descending
      const sorted = [...s6.intervention_results].sort(
        (a, b) => Math.abs(b.effect_size ?? 0) - Math.abs(a.effect_size ?? 0),
      );

      lines.push(section(3, "Treatment Ranking"));
      lines.push("");

      const txRows = sorted.map((t) => {
        const draws = t.posterior_draws;
        let ci = "—";
        if (draws && draws.length > 0) {
          const s = [...draws].sort((a, b) => a - b);
          ci = `[${formatNumber(quantile(s, CI_LOWER))}, ${formatNumber(quantile(s, CI_UPPER))}]`;
        }
        return [
          t.treatment,
          t.effect_size != null ? formatNumber(t.effect_size) : "—",
          ci,
          t.prob_positive != null ? formatPercent(t.prob_positive) : "—",
          t.identifiable ? "Yes" : "No",
        ];
      });
      lines.push(
        mdTable(["Treatment", "\u03C4\u0302", "95% CI", "P(\u03C4>0)", "Identifiable"], txRows),
      );
      lines.push("");

      // ASCII posterior histograms per treatment
      for (const t of sorted) {
        if (t.posterior_draws && t.posterior_draws.length > 0) {
          lines.push(fenced(asciiHistogram(t.posterior_draws, { label: `Posterior: ${t.treatment}` })));
          lines.push("");
        }
      }

      // Temporal effects
      const withTemporal = sorted.filter((t) => t.temporal);
      if (withTemporal.length > 0) {
        lines.push(section(3, "Temporal Effects"));
        lines.push("");
        const tempRows = withTemporal.map((t) => {
          const tmp = t.temporal;
          return [
            t.treatment,
            formatNumber(tmp?.effect_1d ?? 0),
            formatNumber(tmp?.effect_7d ?? 0),
            formatNumber(tmp?.effect_30d ?? 0),
            formatNumber(tmp?.peak_effect ?? 0),
            `${formatNumber(tmp?.time_to_peak_days ?? 0, 1)} days`,
          ];
        });
        lines.push(mdTable(["Treatment", "1d", "7d", "30d", "Peak", "Time to Peak"], tempRows));
        lines.push("");
      }

    }
  }

  return lines.join("\n");
}
