import type {
  PriorProposal,
  Stage0Data,
  Stage1aData,
  Stage1bData,
  Stage2Data,
  Stage3Data,
  Stage4Data,
  Stage4bData,
  Stage5aData,
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
  "stage-5a"?: Stage5aData | null;
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

// ── Summary stat helpers for LLM consumption ──────────────────────────

/** Pearson correlation coefficient between two equal-length arrays. */
function pearsonR(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2) return NaN;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? 0 : num / denom;
}

/** Chi-squared statistic for uniformity test. */
function chiSquaredUniformity(counts: number[], expected: number): number {
  return counts.reduce((acc, c) => acc + (c - expected) ** 2 / expected, 0);
}

/** Max |ECDF(x) - x| (Kolmogorov-Smirnov-like stat for uniformity). */
function ksUniformStat(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  let maxDev = 0;
  for (let i = 0; i < n; i++) {
    const ecdf = (i + 1) / n;
    maxDev = Math.max(maxDev, Math.abs(ecdf - sorted[i]));
  }
  return maxDev;
}

/** RMSE between two equal-length arrays. */
function rmse(a: number[], b: number[]): number {
  const n = a.length;
  if (n === 0) return NaN;
  const mse = a.reduce((acc, v, i) => acc + (v - b[i]) ** 2, 0) / n;
  return Math.sqrt(mse);
}

/** Mean absolute error between two equal-length arrays. */
function mae(a: number[], b: number[]): number {
  const n = a.length;
  if (n === 0) return NaN;
  return a.reduce((acc, v, i) => acc + Math.abs(v - b[i]), 0) / n;
}

/** ELBO convergence summary stats. */
function elboStats(losses: number[]): { initial: number; final: number; improvement: number; converged: boolean } {
  const initial = losses[0];
  const final = losses[losses.length - 1];
  const improvement = initial !== 0 ? Math.abs((initial - final) / initial) : 0;
  // Check last 10% of steps for convergence (relative change < 1%)
  const tail = losses.slice(Math.max(0, Math.floor(losses.length * 0.9)));
  const tailRange = tail.length > 1 ? Math.abs(tail[tail.length - 1] - tail[0]) : 0;
  const converged = tail.length > 1 && Math.abs(final) > 0 ? tailRange / Math.abs(final) < 0.01 : false;
  return { initial, final, improvement, converged };
}

/** Render pre-binned counts as a compact horizontal bar chart. */
function asciiBins(binCounts: number[], expected: number, label: string): string {
  if (binCounts.length === 0) return label;
  const maxCount = Math.max(...binCounts);
  const barScale = maxCount > 0 ? 30 / maxCount : 0;
  const lines: string[] = [label];
  for (let i = 0; i < binCounts.length; i++) {
    const bar = "\u2588".repeat(Math.round(binCounts[i] * barScale));
    lines.push(`  [${String(i + 1).padStart(2)}] ${bar} ${binCounts[i]}`);
  }
  lines.push(`  expected \u2248 ${Math.round(expected)}/bin`);
  return lines.join("\n");
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
    lines.push(`- **Columns**: ${s0.n_columns}`);
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

    if (s0.column_descriptions.length > 0) {
      lines.push(section(3, "Column Descriptions"));
      lines.push("");
      const colDescRows = s0.column_descriptions.map((c) => [c.name, c.dtype, c.description]);
      lines.push(mdTable(["Column", "Type", "Description"], colDescRows));
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
        status?.method ?? "\u2014",
        status?.estimand ?? "\u2014",
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

    // Combined extractions sample
    if (s2.combined_extractions_sample && s2.combined_extractions_sample.length > 0) {
      lines.push(section(3, "Extractions Sample"));
      lines.push("");
      const extRows = s2.combined_extractions_sample.slice(0, 20).map((e) => [
        e.indicator,
        e.tick ?? "\u2014",
        String(e.value ?? "\u2014"),
      ]);
      lines.push(mdTable(["Indicator", "Tick", "Value"], extRows));
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

    const indicatorEntries = Object.entries(s3.indicators ?? {});
    const indicatorIssues = indicatorEntries.flatMap(([, audit]) => audit?.validation.issues ?? []);
    const datasetIssues = s3.dataset_issues ?? [];
    const allIssues = [...indicatorIssues, ...datasetIssues];

    if (!s3.is_valid) {
      lines.push("> **GATE BLOCKED**: Data validation failed.");
      lines.push("");
    }

    if (indicatorEntries.length > 0) {
      lines.push(section(3, "Indicator Audits"));
      lines.push("");

      const issueCounts = new Map<string, { errors: number; warnings: number }>();
      for (const issue of indicatorIssues) {
        if (!issue.indicator) continue;
        const counts = issueCounts.get(issue.indicator) ?? { errors: 0, warnings: 0 };
        if (issue.severity === "error") counts.errors++;
        else if (issue.severity === "warning") counts.warnings++;
        issueCounts.set(issue.indicator, counts);
      }

      const healthRows = indicatorEntries.map(([indicator, audit]) => {
        const profile = audit?.profile;
        const counts = issueCounts.get(indicator);
        const issueStr = counts
          ? [counts.errors > 0 ? `${counts.errors}E` : "", counts.warnings > 0 ? `${counts.warnings}W` : ""]
              .filter(Boolean)
              .join(" ") || "\u2014"
          : "\u2014";
        return [
          indicator,
          issueStr,
          profile?.n_obs != null ? String(profile.n_obs) : "\u2014",
          profile?.mean != null ? formatNumber(profile.mean) : "\u2014",
          profile?.variance != null ? formatNumber(profile.variance) : "\u2014",
          profile?.time_coverage_ratio != null ? formatPercent(profile.time_coverage_ratio) : "\u2014",
          profile?.max_gap_ratio != null ? formatPercent(profile.max_gap_ratio) : "\u2014",
          profile?.dtype_violations != null ? String(profile.dtype_violations) : "\u2014",
          profile?.duplicate_pct != null ? formatPercent(profile.duplicate_pct) : "\u2014",
          profile?.arithmetic_sequence_detected ? "Yes" : "No",
        ];
      });
      lines.push(
        mdTable(
          [
            "Indicator",
            "Issues",
            "N obs",
            "Mean",
            "Variance",
            "Coverage",
            "Max Gap",
            "Type Violations",
            "Duplicates",
            "Arith. Seq.",
          ],
          healthRows,
        ),
      );
      lines.push("");
    }

    if (allIssues.length > 0) {
      lines.push(section(3, "Issues"));
      lines.push("");
      for (const issue of allIssues) {
        const prefix = issue.indicator ? `${issue.indicator}: ` : "";
        lines.push(`- ${prefix}${issue.message} (${issue.severity})`);
      }
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
    const s1bForS4 = data["stage-1b"];
    const indMap = s1bForS4
      ? Object.fromEntries(s1bForS4.causal_spec.measurement.indicators.map((ind) => [ind.name, ind.construct_name]))
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

      // Measurement table with sources
      lines.push(section(3, "Measurement Model"));
      lines.push("");
      const measRows = s4.model_spec.likelihoods.map((l) => {
        const sourceLinks = (l.sources ?? [])
          .map((src) => (src.url ? `[${src.title}](${src.url})` : src.title))
          .join("; ");
        return [
          l.variable,
          l.distribution,
          l.link,
          l.reasoning,
          sourceLinks || "\u2014",
        ];
      });
      lines.push(mdTable(["Variable", "Distribution", "Link", "Reasoning", "Sources"], measRows));
      lines.push("");

      // Search context for likelihoods
      const withLikContext = s4.model_spec.likelihoods.filter((l) => l.search_context);
      if (withLikContext.length > 0) {
        lines.push(section(4, "Likelihood Search Context"));
        lines.push("");
        for (const l of withLikContext) {
          lines.push(`**${l.variable}**: ${l.search_context}`);
          lines.push("");
        }
      }
    }

    // Build initial state priors
    const states = stateNames(s4.model_spec.parameters);
    const t0Priors: PriorProposal[] = [];
    for (const st of states) {
      t0Priors.push({
        parameter: `t0_mean_${st}`,
        distribution: "Normal",
        params: { mu: 0, sigma: 2 },
        sources: [],
        reasoning: `Default weakly informative prior for the initial state mean of ${textify(st)}.`,
      });
      t0Priors.push({
        parameter: `t0_sd_${st}`,
        distribution: "HalfNormal",
        params: { sigma: 2 },
        sources: [],
        reasoning: `Default weakly informative prior for the initial state standard deviation of ${textify(st)}.`,
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
          .map((src) => (src.url ? `[${src.title}](${src.url})` : src.title))
          .join("; ");
        return [
          `$${paramSymbol(p.parameter)}$`,
          `${p.distribution}(${params})`,
          sourceLinks || "\u2014",
          p.reasoning,
        ];
      });
      lines.push(mdTable(["Parameter", "Prior", "Sources", "Reasoning"], priorRows));
      lines.push("");
    }

    // Validation retries
    if (s4.validation_retries && s4.validation_retries.length > 0) {
      lines.push(section(3, "Validation Retries"));
      lines.push("");
      for (const r of s4.validation_retries) {
        lines.push(`**Attempt ${r.attempt}**: Failed params: ${r.failed_params.join(", ")}`);
        lines.push(`> ${r.feedback}`);
        lines.push("");
      }
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
        `> **GATE BLOCKED**: T-Rule violated \u2014 ${pid.t_rule.n_free_params} free parameters but only ${pid.t_rule.n_moments} moment conditions.`,
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
        p.contraction_ratio != null ? formatNumber(p.contraction_ratio) : "\u2014",
      ]);
      lines.push(mdTable(["Parameter", "Classification", "Contraction Ratio"], classRows));
      lines.push("");
    }

    // Sensitivity analysis
    const sa = pid.sensitivity_analysis;
    if (sa) {
      lines.push(section(3, "Sensitivity Analysis"));
      lines.push("");
      lines.push(`- **Condition number**: ${formatNumber(sa.condition_number)}`);
      lines.push(`- **Parameters**: ${sa.n_parameters}`);
      lines.push(`- **Draws**: ${sa.n_draws}`);
      lines.push("");

      if (sa.per_parameter.length > 0) {
        const saRows = sa.per_parameter.map((p) => [
          p.parameter,
          formatNumber(p.sensitivity_norm),
          formatNumber(p.effective_sv),
          p.sv_status,
          formatNumber(p.normalized_effective_sv),
          p.normalized_sv_status,
        ]);
        lines.push(
          mdTable(
            ["Parameter", "Sensitivity Norm", "Effective SV", "SV Status", "Normalized SV", "Norm. Status"],
            saRows,
          ),
        );
        lines.push("");
      }
    }

    lines.push("---");
    lines.push("");
  }

  // --- Stage 5a: SVI Preflight ---
  const s5a = data["stage-5a"];
  if (s5a) {
    lines.push(section(2, `Stage 5a: ${STAGES[7].label}`));
    lines.push(`> ${STAGES[7].description}`);
    lines.push("");

    // SVI diagnostics
    if (s5a.svi_diagnostics) {
      const losses = s5a.svi_diagnostics.elbo_losses;
      lines.push(section(3, "SVI / ELBO Convergence"));
      lines.push("");
      lines.push(
        fenced(
          asciiDensity(
            losses.map((_, i) => i),
            losses,
            { label: "ELBO loss over optimization steps", height: 10 },
          ),
        ),
      );
      if (losses.length >= 2) {
        const es = elboStats(losses);
        lines.push(`Initial loss: ${formatNumber(es.initial, 1)}, Final loss: ${formatNumber(es.final, 1)}, Improvement: ${formatPercent(es.improvement)}, Converged: ${es.converged ? "Yes" : "No"}`);
      }
      lines.push("");
    }

    // Posterior marginals
    if (s5a.posterior_marginals && s5a.posterior_marginals.length > 0) {
      lines.push(section(3, "Posterior Marginals"));
      lines.push("");
      for (const m of s5a.posterior_marginals) {
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

    // Posterior pairs
    if (s5a.posterior_pairs && s5a.posterior_pairs.length > 0) {
      lines.push(section(3, "Posterior Pairs"));
      lines.push("");
      for (const pair of s5a.posterior_pairs) {
        const nDiv = pair.divergent ? pair.divergent.filter(Boolean).length : 0;
        const nTotal = pair.x_values.length;
        const r = pearsonR(pair.x_values, pair.y_values);
        const points = pair.x_values.map((x, i) => ({ x, y: pair.y_values[i] }));
        lines.push(
          fenced(
            asciiScatter(points, {
              label: `${pair.param_x} vs ${pair.param_y}${nDiv > 0 ? ` (${nDiv} divergent)` : ""}`,
              height: 15,
              width: 50,
            }),
          ),
        );
        lines.push(`Pearson r: ${formatNumber(r)}${nDiv > 0 ? `, Divergent: ${nDiv}/${nTotal} (${formatPercent(nDiv / nTotal)})` : ""}`);
        lines.push("");
      }
    }

    // Inference metadata
    lines.push(
      `*SVI Preflight: ${s5a.inference_metadata.method} | ${s5a.inference_metadata.n_samples} samples | ${s5a.inference_metadata.duration_seconds.toFixed(1)}s*`,
    );
    lines.push("");
    lines.push("---");
    lines.push("");
  }

  // --- Stage 5b: Inference & Diagnostics ---
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

      // Per-parameter convergence table (with MCSE)
      if (mcmc.per_parameter.length > 0) {
        lines.push(section(4, "Convergence"));
        lines.push("");
        const convRows = mcmc.per_parameter.map((p) => {
          const rhat = Array.isArray(p.r_hat) ? p.r_hat.map((v) => formatNumber(v)).join(", ") : formatNumber(p.r_hat);
          const ess = Array.isArray(p.ess_bulk) ? p.ess_bulk.map((v) => formatNumber(v, 0)).join(", ") : formatNumber(p.ess_bulk, 0);
          const essTail = p.ess_tail
            ? (Array.isArray(p.ess_tail) ? p.ess_tail.map((v) => formatNumber(v, 0)).join(", ") : formatNumber(p.ess_tail, 0))
            : "\u2014";
          const mcse = p.mcse_mean
            ? (Array.isArray(p.mcse_mean) ? p.mcse_mean.map((v) => formatNumber(v)).join(", ") : formatNumber(p.mcse_mean))
            : "\u2014";
          return [p.parameter, rhat, ess, essTail, mcse];
        });
        lines.push(mdTable(["Parameter", "R-hat", "ESS bulk", "ESS tail", "MCSE"], convRows));
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

      // Rank histograms
      if (mcmc.rank_histograms && mcmc.rank_histograms.length > 0) {
        lines.push(section(4, "Rank Histograms"));
        lines.push("");
        for (const rh of mcmc.rank_histograms) {
          // Sum counts across chains
          const totalCounts = new Array<number>(rh.n_bins).fill(0);
          for (const chain of rh.chains) {
            for (let i = 0; i < chain.counts.length; i++) {
              totalCounts[i] += chain.counts[i];
            }
          }
          const totalExpected = rh.expected_per_bin * rh.chains.length;
          lines.push(fenced(asciiBins(totalCounts, totalExpected, rh.parameter)));
          const chi2 = chiSquaredUniformity(totalCounts, totalExpected);
          const maxDev = Math.max(...totalCounts.map((c) => Math.abs(c - totalExpected) / totalExpected));
          lines.push(`Chi-squared: ${formatNumber(chi2, 1)}, Max deviation: ${formatPercent(maxDev)}, Uniformity: ${maxDev < 0.2 ? "OK" : "Concern"}`);
          lines.push("");
        }
      }

      // Energy diagnostics
      if (mcmc.energy) {
        lines.push(section(4, "Energy Diagnostics"));
        lines.push("");
        lines.push(`**BFMI per chain**: ${mcmc.energy.bfmi.map((v) => formatNumber(v)).join(", ")}${mcmc.energy.bfmi.some((v) => v < 0.3) ? " (values < 0.3 indicate concern)" : ""}`);
        lines.push("");
        if (mcmc.energy.energy_hist.bin_centers.length > 0) {
          lines.push(
            fenced(
              asciiDensity(mcmc.energy.energy_hist.bin_centers, mcmc.energy.energy_hist.density, {
                label: "Marginal Energy",
                height: 8,
                width: 50,
              }),
            ),
          );
          lines.push("");
        }
        if (mcmc.energy.energy_transition_hist.bin_centers.length > 0) {
          lines.push(
            fenced(
              asciiDensity(mcmc.energy.energy_transition_hist.bin_centers, mcmc.energy.energy_transition_hist.density, {
                label: "Energy Transition",
                height: 8,
                width: 50,
              }),
            ),
          );
          lines.push("");
        }
      }
    }

    // SVI diagnostics
    if (s5.svi_diagnostics) {
      const losses5b = s5.svi_diagnostics.elbo_losses;
      lines.push(section(3, "SVI / ELBO Convergence"));
      lines.push("");
      lines.push(
        fenced(
          asciiDensity(
            losses5b.map((_, i) => i),
            losses5b,
            { label: "ELBO loss over optimization steps", height: 10 },
          ),
        ),
      );
      if (losses5b.length >= 2) {
        const es = elboStats(losses5b);
        lines.push(`Initial loss: ${formatNumber(es.initial, 1)}, Final loss: ${formatNumber(es.final, 1)}, Improvement: ${formatPercent(es.improvement)}, Converged: ${es.converged ? "Yes" : "No"}`);
      }
      lines.push("");
    }

    // SMC diagnostics
    if (s5.smc_diagnostics) {
      const smc = s5.smc_diagnostics;
      lines.push(section(3, "SMC Diagnostics"));
      lines.push("");
      lines.push(`- **Particles**: ${smc.n_particles}`);
      lines.push(`- **Levels**: ${smc.n_levels}`);
      lines.push("");

      if (smc.beta_schedule.length > 0 && smc.ess_history.length > 0) {
        lines.push(section(4, "Tempering Schedule & ESS"));
        lines.push("");
        const smcRows = smc.beta_schedule.map((beta, i) => [
          String(i),
          formatNumber(beta),
          i < smc.ess_history.length ? formatNumber(smc.ess_history[i], 0) : "\u2014",
          i < smc.accept_rates.length ? formatPercent(smc.accept_rates[i]) : "\u2014",
        ]);
        lines.push(mdTable(["Level", "\u03B2", "ESS", "Accept Rate"], smcRows));
        lines.push("");
      }

      // ESS over levels as ASCII chart
      if (smc.ess_history.length >= 2) {
        lines.push(
          fenced(
            asciiDensity(
              smc.ess_history.map((_, i) => i),
              smc.ess_history,
              { label: "ESS over tempering levels", height: 8, width: 50 },
            ),
          ),
        );
        const minEss = Math.min(...smc.ess_history);
        const meanEss = smc.ess_history.reduce((a, b) => a + b, 0) / smc.ess_history.length;
        const finalEss = smc.ess_history[smc.ess_history.length - 1];
        lines.push(`Min ESS: ${formatNumber(minEss, 0)}, Mean ESS: ${formatNumber(meanEss, 0)}, Final ESS: ${formatNumber(finalEss, 0)}`);
        lines.push("");
      }
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

    // PPC overlays
    if (s5.ppc.overlays.length > 0) {
      lines.push(section(4, "Posterior Predictive Overlays"));
      lines.push("");
      for (const overlay of s5.ppc.overlays) {
        // Extract indices where both observed and median are valid numbers
        const validIdx: number[] = [];
        for (let i = 0; i < overlay.observed.length; i++) {
          if (overlay.observed[i] != null && overlay.median[i] != null) validIdx.push(i);
        }
        if (validIdx.length === 0) continue;

        const obsValues = validIdx.map((i) => overlay.observed[i] as number);
        const medValues = validIdx.map((i) => overlay.median[i]);

        lines.push(
          fenced(
            asciiMultiLine([obsValues, medValues], {
              label: `${overlay.variable} (\u2022 observed, \u25E6 median)`,
              height: 10,
              width: 60,
            }),
          ),
        );

        // 95% CI coverage
        let inBand = 0;
        for (const i of validIdx) {
          const obs = overlay.observed[i] as number;
          if (obs >= overlay.q025[i] && obs <= overlay.q975[i]) inBand++;
        }
        const overlayRmse = rmse(obsValues, medValues);
        const overlayMae = mae(obsValues, medValues);
        const overlayR = pearsonR(obsValues, medValues);
        lines.push(`95% CI coverage: ${formatPercent(inBand / validIdx.length)} (${inBand}/${validIdx.length}), RMSE: ${formatNumber(overlayRmse)}, MAE: ${formatNumber(overlayMae)}, Pearson r: ${formatNumber(overlayR)}`);
        lines.push("");
      }
    }

    // PPC test statistics
    if (s5.ppc.test_stats.length > 0) {
      lines.push(section(4, "Test Statistics"));
      lines.push("");
      const testRows = s5.ppc.test_stats.map((t) => {
        const n = t.rep_values.length;
        const pValue = n > 0 ? t.rep_values.filter((v) => v >= t.observed_value).length / n : NaN;
        return [
          t.variable,
          t.stat_name,
          formatNumber(t.observed_value),
          formatNumber(pValue),
          pValue < 0.05 || pValue > 0.95 ? "Fail" : "Pass",
        ];
      });
      lines.push(mdTable(["Variable", "Statistic", "Observed", "p(rep \u2265 obs)", "Result"], testRows));
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
      lines.push(`- **Observation unit**: ${loo.observation_unit}`);
      if (loo.n_bad_k != null) {
        lines.push(`- **Bad Pareto k**: ${loo.n_bad_k}`);
      }
      lines.push("");

      // LOO-PIT histogram
      if (loo.loo_pit && loo.loo_pit.length > 0) {
        lines.push(section(4, "LOO-PIT"));
        lines.push("");
        lines.push(fenced(asciiHistogram(loo.loo_pit, { label: "LOO-PIT (should be uniform)", nBins: 10 })));
        const pitMean = loo.loo_pit.reduce((a, b) => a + b, 0) / loo.loo_pit.length;
        const pitStd = Math.sqrt(loo.loo_pit.reduce((a, v) => a + (v - pitMean) ** 2, 0) / loo.loo_pit.length);
        const ks = ksUniformStat(loo.loo_pit);
        lines.push(`Mean: ${formatNumber(pitMean)} (ideal: 0.500), Std: ${formatNumber(pitStd)} (ideal: 0.289), KS stat: ${formatNumber(ks)}, Calibration: ${ks < 0.1 ? "Good" : ks < 0.2 ? "Fair" : "Poor"}`);
        lines.push("");
      }

      // Pareto k per observation
      if (loo.pareto_k && loo.pareto_k.length > 0) {
        lines.push(section(4, "Pareto k Diagnostics"));
        lines.push("");
        const nBadK07 = loo.pareto_k.filter((k) => k > 0.7).length;
        const nWarnK05 = loo.pareto_k.filter((k) => k > 0.5 && k <= 0.7).length;
        lines.push(`- **k > 0.7 (fail)**: ${nBadK07}`);
        lines.push(`- **0.5 < k \u2264 0.7 (warn)**: ${nWarnK05}`);
        lines.push(`- **k \u2264 0.5 (ok)**: ${loo.pareto_k.length - nBadK07 - nWarnK05}`);
        lines.push("");
        lines.push(fenced(asciiHistogram(loo.pareto_k, { label: "Pareto k distribution", nBins: 15 })));
        lines.push("");
      }
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
        p.psis_k_hat != null ? formatNumber(p.psis_k_hat) : "\u2014",
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

    // Posterior pairs
    if (s5.posterior_pairs && s5.posterior_pairs.length > 0) {
      lines.push(section(3, "Posterior Pairs"));
      lines.push("");
      for (const pair of s5.posterior_pairs) {
        const nDiv = pair.divergent ? pair.divergent.filter(Boolean).length : 0;
        const nTotal = pair.x_values.length;
        const r = pearsonR(pair.x_values, pair.y_values);
        const points = pair.x_values.map((x, i) => ({ x, y: pair.y_values[i] }));
        lines.push(
          fenced(
            asciiScatter(points, {
              label: `${pair.param_x} vs ${pair.param_y}${nDiv > 0 ? ` (${nDiv} divergent)` : ""}`,
              height: 15,
              width: 50,
            }),
          ),
        );
        lines.push(`Pearson r: ${formatNumber(r)}${nDiv > 0 ? `, Divergent: ${nDiv}/${nTotal} (${formatPercent(nDiv / nTotal)})` : ""}`);
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
        let ci = "\u2014";
        if (draws && draws.length > 0) {
          const sorted_draws = [...draws].sort((a, b) => a - b);
          ci = `[${formatNumber(quantile(sorted_draws, CI_LOWER))}, ${formatNumber(quantile(sorted_draws, CI_UPPER))}]`;
        }
        const warnings: string[] = [];
        if (!t.identifiable) warnings.push("non-identifiable");
        if (t.prior_sensitivity_warning) warnings.push("prior-sensitive");
        const statusStr = warnings.length > 0 ? warnings.join(", ") : "ok";
        return [
          t.treatment,
          t.effect_size != null ? formatNumber(t.effect_size) : "\u2014",
          ci,
          t.prob_positive != null ? formatPercent(t.prob_positive) : "\u2014",
          t.identifiable ? "Yes" : "No",
          statusStr,
        ];
      });
      lines.push(
        mdTable(["Treatment", "\u03C4\u0302", "95% CI", "P(\u03C4>0)", "Identifiable", "Status"], txRows),
      );
      lines.push("");

      // Prior sensitivity warnings
      const withPriorWarn = sorted.filter((t) => t.prior_sensitivity_warning);
      if (withPriorWarn.length > 0) {
        lines.push(section(4, "Prior Sensitivity Warnings"));
        lines.push("");
        for (const t of withPriorWarn) {
          lines.push(`- **${t.treatment}**: ${t.prior_sensitivity_warning}`);
        }
        lines.push("");
      }

      // ASCII posterior histograms per treatment
      for (const t of sorted) {
        if (t.posterior_draws && t.posterior_draws.length > 0) {
          lines.push(fenced(asciiHistogram(t.posterior_draws, { label: `Posterior: ${t.treatment}` })));
          lines.push("");
        }
      }

      // Manifest effects
      const withManifest = sorted.filter((t) => t.manifest_effects && Object.keys(t.manifest_effects).length > 0);
      if (withManifest.length > 0) {
        lines.push(section(3, "Manifest Effects"));
        lines.push("");
        const manifestRows: string[][] = [];
        for (const t of withManifest) {
          for (const [indicator, effect] of Object.entries(t.manifest_effects ?? {})) {
            manifestRows.push([t.treatment, indicator, formatNumber(effect ?? 0)]);
          }
        }
        lines.push(mdTable(["Treatment", "Indicator", "Effect"], manifestRows));
        lines.push("");
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
