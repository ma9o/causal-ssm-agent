import { describe, expect, it } from "vitest";

// The helper functions are not exported, so we test them through the module.
// We can test the exported generateMarkdown with minimal stage data.
import { type AllStageData, generateMarkdown } from "./generate-markdown";

describe("generateMarkdown", () => {
  it("generates header with run ID", () => {
    const data: AllStageData = {};
    const result = generateMarkdown(data, "test-run-123");
    expect(result).toContain("# Causal Inference Pipeline Report");
    expect(result).toContain("`test-run-123`");
    expect(result).toContain("**Generated**:");
  });

  it("generates empty report without crashing", () => {
    const data: AllStageData = {};
    const result = generateMarkdown(data, "empty");
    expect(typeof result).toBe("string");
    expect(result.length).toBeGreaterThan(0);
  });

  it("includes stage 0 section when data is present", () => {
    const data: AllStageData = {
      "stage-0": {
        outcome: "success",
        n_records: 100,
        n_columns: 2,
        date_range: { start: "2024-01-01", end: "2024-12-31" },
        sample: [{ timestamp: "2024-01-01", value: "42" }],
        column_descriptions: [],
      },
    };
    const result = generateMarkdown(data, "run-1");
    expect(result).toContain("Stage 0");
    expect(result).toContain("100");
  });

  it("includes stage 0 n_columns and column descriptions", () => {
    const data: AllStageData = {
      "stage-0": {
        outcome: "success",
        n_records: 100,
        n_columns: 5,
        date_range: { start: "2024-01-01", end: "2024-12-31" },
        sample: [],
        column_descriptions: [
          { name: "timestamp", dtype: "datetime", description: "When the event occurred" },
          { name: "heart_rate", dtype: "float", description: "Heart rate in BPM" },
        ],
      },
    };
    const result = generateMarkdown(data, "run-cols");
    expect(result).toContain("**Columns**: 5");
    expect(result).toContain("Column Descriptions");
    expect(result).toContain("timestamp");
    expect(result).toContain("Heart rate in BPM");
  });

  it("handles null stage data gracefully", () => {
    const data: AllStageData = {
      "stage-0": null,
      "stage-1a": null,
    };
    const result = generateMarkdown(data, "null-test");
    expect(typeof result).toBe("string");
  });

  it("includes stage 6 treatment effects section", () => {
    const data: AllStageData = {
      "stage-6": {
        outcome: "success",
        intervention_results: [
          {
            treatment: "exercise",
            effect_size: 0.35,
            identifiable: true,
            prob_positive: 0.92,
            posterior_draws: [0.1, 0.2, 0.3, 0.4, 0.5],
          },
        ],
        inference_metadata: {
          method: "svi",
          n_samples: 1000,
          duration_seconds: 30.5,
        },
      } as AllStageData["stage-6"],
    };
    const result = generateMarkdown(data, "run-effects");
    expect(result).toContain("exercise");
    expect(result).toContain("Treatment");
  });

  it("includes multiple stages together", () => {
    const data: AllStageData = {
      "stage-0": {
        outcome: "success",
        n_records: 50,
        n_columns: 1,
        date_range: { start: "2024-01-01", end: "2024-06-30" },
        sample: [],
        column_descriptions: [],
      },
      "stage-3": {
        outcome: "success",
        is_valid: true,
        indicators: {},
        dataset_issues: [],
      } as AllStageData["stage-3"],
    };
    const result = generateMarkdown(data, "multi-stage");
    expect(result).toContain("Stage 0");
    expect(result).toContain("Stage 3");
  });

  it("includes first-pass RB latent and observation assignments", () => {
    const data: AllStageData = {
      "stage-4b": {
        outcome: "success",
        parametric_id: { checked: true },
        inference_structure: {
          likelihood_path: "composed",
          auto_method: "laplace_em",
          first_pass_rb: {
            status: "active",
            inactive_reason: null,
            latent_variables: [
              { name: "g0", method: "kalman" },
              { name: "s0", method: "particle" },
            ],
            obs_variables: [
              { name: "yg0", method: "kalman" },
              { name: "ys0", method: "particle" },
            ],
          },
        },
      } as AllStageData["stage-4b"],
    };
    const result = generateMarkdown(data, "rb-stage");
    expect(result).toContain("Inference Structure");
    expect(result).toContain("Likelihood path");
    expect(result).toContain("Latents (Kalman)");
    expect(result).toContain("Observed Channels (Particle-side)");
    expect(result).toContain("yg0");
    expect(result).toContain("ys0");
  });

  it("contains date in header", () => {
    const result = generateMarkdown({}, "test");
    // Should have ISO-like date format
    expect(result).toMatch(/\d{4}/);
  });

  it("includes stage 1a constructs and edges", () => {
    const data: AllStageData = {
      "stage-1a": {
        outcome: "success",
        latent_model: {
          constructs: [
            {
              name: "exercise",
              description: "Physical activity",
              role: "exogenous",
              is_outcome: false,
              temporal_status: "time_varying",
            },
            {
              name: "blood_pressure",
              description: "BP measurement",
              role: "endogenous",
              is_outcome: true,
              temporal_status: "time_varying",
            },
          ],
          edges: [
            {
              cause: "exercise",
              effect: "blood_pressure",
              lagged: true,
              description: "Regular exercise lowers BP",
            },
          ],
        },
      } as AllStageData["stage-1a"],
    };
    const result = generateMarkdown(data, "run-1a");
    expect(result).toContain("exercise");
    expect(result).toContain("blood_pressure");
    expect(result).toContain("Stage 1a");
    expect(result).toContain("Constructs");
  });

  it("includes stage 2 combined extractions sample", () => {
    const data: AllStageData = {
      "stage-2": {
        outcome: "success",
        workers: [
          { worker_id: 0, status: "completed" as const, n_extractions: 10, n_windows: 5 },
        ],
        per_indicator_counts: { heart_rate: 10 },
        combined_extractions_sample: [
          { indicator: "heart_rate", value: 72, anchor_time: "2024-01-01T00:00:00Z" },
          { indicator: "heart_rate", value: 75, anchor_time: "2024-01-02T00:00:00Z" },
        ],
      } as AllStageData["stage-2"],
    };
    const result = generateMarkdown(data, "run-2");
    expect(result).toContain("Extractions Sample");
    expect(result).toContain("heart_rate");
    expect(result).toContain("72");
  });

  it("includes stage 3 validation issues with issue counts and arith seq", () => {
    const data: AllStageData = {
      "stage-3": {
        outcome: "success",
        is_valid: false,
        indicators: {
          heart_rate: {
            profile: {
              measurement_dtype: "continuous",
              n_obs: 100,
              mean: 72,
              std: 10.2,
              min: 50,
              max: 110,
              q25: 65,
              q50: 72,
              q75: 80,
              variance: 104.04,
              time_coverage_ratio: 0.95,
              max_gap_ratio: 0.02,
              dtype_violations: 0,
              duplicate_pct: 0.01,
              arithmetic_sequence_detected: false,
              n_unparseable_timestamps: 0,
              zero_fraction: 0,
              is_nonnegative: true,
              is_unit_interval: false,
              looks_integer_valued: true,
              variance_to_mean_ratio: 1.44,
            },
            validation: {
              issues: [
                { indicator: "heart_rate", issue_type: "missing_data", severity: "error" as const, message: "Missing" },
                { indicator: "heart_rate", issue_type: "low_variance", severity: "warning" as const, message: "Low var" },
              ],
              checks: { n_obs: "error", variance: "warning" },
            },
          },
          steps: {
            profile: {
              measurement_dtype: "count",
              n_obs: 50,
              mean: 3000,
              std: 3.2,
              min: 2995,
              max: 3005,
              q25: 2998,
              q50: 3000,
              q75: 3002,
              variance: 10.0,
              time_coverage_ratio: 0.5,
              max_gap_ratio: 0.1,
              dtype_violations: 2,
              duplicate_pct: 0.0,
              arithmetic_sequence_detected: true,
              n_unparseable_timestamps: 0,
              zero_fraction: 0,
              is_nonnegative: true,
              is_unit_interval: false,
              looks_integer_valued: true,
              variance_to_mean_ratio: 0.0033,
            },
            validation: {
              issues: [
                { indicator: "steps", issue_type: "dtype", severity: "error" as const, message: "Bad type" },
              ],
              checks: { dtype_violations: "error", arithmetic_sequence_detected: "warning" },
            },
          },
        },
        dataset_issues: [],
      } as AllStageData["stage-3"],
    };
    const result = generateMarkdown(data, "run-3");
    expect(result).toContain("Stage 3");
    expect(result).toContain("PIPELINE STOPPED");
    expect(result).toContain("heart_rate");
    expect(result).toContain("1E 1W"); // heart_rate has 1 error, 1 warning
    expect(result).toContain("1E"); // steps has 1 error
    expect(result).toContain("Arith. Seq.");
    expect(result).toContain("Yes"); // steps has arithmetic_sequence_detected
  });

  it("includes stage 4 measurement sources", () => {
    const data: AllStageData = {
      "stage-4": {
        outcome: "success",
        model_spec: {
          parameters: [],
          likelihoods: [
            {
              variable: "heart_rate",
              distribution: "gaussian",
              link: "identity",
              reasoning: "Continuous measurement",
              sources: [{ title: "Study A", url: "https://example.com/a", snippet: "HR is gaussian" }],
            },
          ],
        },
        authored_priors: {},
        resolved_priors: [],
      } as AllStageData["stage-4"],
    };
    const result = generateMarkdown(data, "run-4");
    expect(result).toContain("Sources");
    expect(result).toContain("[Study A](https://example.com/a)");
  });

  it("includes stage 4b sensitivity analysis", () => {
    const data: AllStageData = {
      "stage-4b": {
        outcome: "success",
        parametric_id: {
          checked: true,
          t_rule: {
            n_free_params: 5,
            n_manifest: 3,
            n_timepoints: 10,
            n_moments: 15,
            satisfies: true,
            param_counts: {},
          },
          sensitivity_analysis: {
            condition_number: 42.5,
            n_parameters: 5,
            n_draws: 1000,
            n_observations: 100,
            singular_values: [1.0, 0.5, 0.1],
            per_parameter: [
              {
                parameter: "rho_x",
                sensitivity_norm: 0.8,
                effective_sv: 0.5,
                sv_status: "pass" as const,
                normalized_effective_sv: 12.0,
                normalized_sv_status: "pass" as const,
                identifiable: true,
              },
              {
                parameter: "sigma_y",
                sensitivity_norm: 0.01,
                effective_sv: 0.0001,
                sv_status: "fail" as const,
                normalized_effective_sv: 0.5,
                normalized_sv_status: "warn" as const,
                identifiable: false,
              },
            ],
          },
        },
      } as AllStageData["stage-4b"],
    };
    const result = generateMarkdown(data, "run-4b");
    expect(result).toContain("Sensitivity Analysis");
    expect(result).toContain("Condition number");
    expect(result).toContain("42.500");
    expect(result).toContain("rho_x");
    expect(result).toContain("sigma_y");
    expect(result).toContain("Effective SV");
    expect(result).toContain("fail");
  });

  it("surfaces stage 4b t-rule warnings as warnings rather than pipeline stops", () => {
    const data: AllStageData = {
      "stage-4b": {
        outcome: "warn",
        parametric_id: {
          checked: true,
          t_rule: {
            n_free_params: 12,
            n_manifest: 3,
            n_timepoints: 8,
            n_moments: 10,
            satisfies: false,
            param_counts: {},
          },
          error:
            "T-rule warning: 12 free params > conservative lower-bound 10 moment conditions. This screen is warning-only and does not halt inference.",
        },
      } as AllStageData["stage-4b"],
    };

    const result = generateMarkdown(data, "run-4b-warning");

    expect(result).toContain("**WARNING**");
    expect(result).toContain("T-Rule screen failed");
    expect(result).toContain("Lower-bound moment conditions");
    expect(result).not.toContain("PIPELINE STOPPED");
  });

  it("includes stage 5a SVI preflight", () => {
    const data: AllStageData = {
      "stage-5a": {
        outcome: "success",
        inference_metadata: {
          method: "svi",
          n_samples: 500,
          duration_seconds: 15.2,
        },
        svi_diagnostics: {
          elbo_losses: [100, 80, 60, 50, 45, 42, 41, 40.5, 40.2, 40.1],
        },
        posterior_marginals: [
          {
            parameter: "rho_x",
            x_values: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
            density: [0.1, 0.3, 0.8, 1.0, 0.5, 0.1],
            mean: 0.5,
            sd: 0.15,
            hdi_3: 0.2,
            hdi_97: 0.8,
          },
        ],
        posterior_pairs: [
          {
            param_x: "rho_x",
            param_y: "sigma_x",
            x_values: [0.4, 0.5, 0.6, 0.45, 0.55],
            y_values: [0.1, 0.12, 0.08, 0.11, 0.09],
          },
        ],
      } as AllStageData["stage-5a"],
    };
    const result = generateMarkdown(data, "run-5a");
    expect(result).toContain("Stage 5a");
    expect(result).toContain("SVI Preflight");
    expect(result).toContain("ELBO loss");
    expect(result).toContain("rho_x");
    expect(result).toContain("Posterior Marginals");
    expect(result).toContain("Posterior Pairs");
    expect(result).toContain("rho_x vs sigma_x");
    expect(result).toContain("Pearson r:");
    expect(result).toContain("15.2s");
    // ELBO convergence stats
    expect(result).toContain("Initial loss:");
    expect(result).toContain("Final loss:");
    expect(result).toContain("Converged:");
  });

  it("handles stage 5b MCMC diagnostics with MCSE", () => {
    const data: AllStageData = {
      "stage-5b": {
        outcome: "success",
        inference_metadata: {
          method: "nuts",
          n_samples: 2000,
          duration_seconds: 120,
        },
        mcmc_diagnostics: {
          num_divergences: 0,
          divergence_rate: 0,
          tree_depth_mean: 5.2,
          tree_depth_max: 8,
          accept_prob_mean: 0.85,
          num_chains: 4,
          num_samples: 2000,
          per_parameter: [
            {
              parameter: "drift_diag_0",
              r_hat: 1.001,
              ess_bulk: 1800,
              ess_tail: 1600,
              mcse_mean: 0.002,
            },
          ],
        },
        ppc: {
          per_variable_warnings: [
            {
              variable: "heart_rate",
              check_type: "calibration" as const,
              passed: true,
              value: 0.12,
              message: "OK",
            },
          ],
          overlays: [],
          test_stats: [],
        },
        power_scaling: [],
      } as AllStageData["stage-5b"],
    };
    const result = generateMarkdown(data, "run-5");
    expect(result).toContain("Stage 5b");
    expect(result).toContain("MCMC Diagnostics");
    expect(result).toContain("drift_diag_0");
    expect(result).toContain("MCSE");
    expect(result).toContain("0.002");
  });

  it("includes stage 5b rank histograms", () => {
    const data: AllStageData = {
      "stage-5b": {
        outcome: "success",
        inference_metadata: { method: "nuts", n_samples: 1000, duration_seconds: 60 },
        mcmc_diagnostics: {
          num_divergences: 0,
          divergence_rate: 0,
          tree_depth_mean: 5,
          tree_depth_max: 7,
          accept_prob_mean: 0.9,
          per_parameter: [],
          rank_histograms: [
            {
              parameter: "rho_x",
              n_bins: 5,
              expected_per_bin: 50,
              chains: [
                { chain: 0, counts: [25, 26, 24, 25, 25] },
                { chain: 1, counts: [24, 25, 26, 25, 25] },
              ],
            },
          ],
        },
        ppc: { per_variable_warnings: [], overlays: [], test_stats: [] },
        power_scaling: [],
      } as AllStageData["stage-5b"],
    };
    const result = generateMarkdown(data, "run-5-rank");
    expect(result).toContain("Rank Histograms");
    expect(result).toContain("rho_x");
    expect(result).toContain("expected");
    expect(result).toContain("Chi-squared:");
    expect(result).toContain("Max deviation:");
    expect(result).toContain("Uniformity:");
  });

  it("includes stage 5b energy diagnostics", () => {
    const data: AllStageData = {
      "stage-5b": {
        outcome: "success",
        inference_metadata: { method: "nuts", n_samples: 1000, duration_seconds: 60 },
        mcmc_diagnostics: {
          num_divergences: 0,
          divergence_rate: 0,
          tree_depth_mean: 5,
          tree_depth_max: 7,
          accept_prob_mean: 0.9,
          per_parameter: [],
          energy: {
            bfmi: [0.85, 0.82],
            energy_hist: { bin_centers: [1, 2, 3, 4, 5], density: [0.1, 0.3, 0.4, 0.15, 0.05] },
            energy_transition_hist: { bin_centers: [0.5, 1, 1.5, 2], density: [0.2, 0.5, 0.25, 0.05] },
          },
        },
        ppc: { per_variable_warnings: [], overlays: [], test_stats: [] },
        power_scaling: [],
      } as AllStageData["stage-5b"],
    };
    const result = generateMarkdown(data, "run-5-energy");
    expect(result).toContain("Energy Diagnostics");
    expect(result).toContain("BFMI per chain");
    expect(result).toContain("0.850");
    expect(result).toContain("Marginal Energy");
    expect(result).toContain("Energy Transition");
  });

  it("includes stage 5b SMC diagnostics", () => {
    const data: AllStageData = {
      "stage-5b": {
        outcome: "success",
        inference_metadata: { method: "smc", n_samples: 1000, duration_seconds: 60 },
        smc_diagnostics: {
          n_particles: 500,
          n_levels: 5,
          beta_schedule: [0.0, 0.25, 0.5, 0.75, 1.0],
          ess_history: [500, 400, 350, 300, 280],
          accept_rates: [1.0, 0.8, 0.7, 0.65, 0.6],
        },
        ppc: { per_variable_warnings: [], overlays: [], test_stats: [] },
        power_scaling: [],
      } as AllStageData["stage-5b"],
    };
    const result = generateMarkdown(data, "run-5-smc");
    expect(result).toContain("SMC Diagnostics");
    expect(result).toContain("500");
    expect(result).toContain("Tempering Schedule");
    expect(result).toContain("Accept Rate");
    expect(result).toContain("ESS over tempering");
    expect(result).toContain("Min ESS:");
    expect(result).toContain("Mean ESS:");
    expect(result).toContain("Final ESS:");
  });

  it("includes stage 5b PPC overlays and test stats", () => {
    const data: AllStageData = {
      "stage-5b": {
        outcome: "success",
        inference_metadata: { method: "nuts", n_samples: 1000, duration_seconds: 60 },
        ppc: {
          per_variable_warnings: [],
          overlays: [
            {
              variable: "heart_rate",
              observed: [70, 72, null, 75, 73],
              q025: [65, 67, 68, 69, 68],
              q25: [68, 70, 71, 72, 71],
              median: [71, 73, 74, 74, 73],
              q75: [74, 76, 77, 76, 75],
              q975: [78, 80, 81, 80, 79],
              spaghetti_draws: [],
            },
          ],
          test_stats: [
            {
              variable: "heart_rate",
              stat_name: "mean" as const,
              observed_value: 72.5,
              rep_values: [71, 72, 73, 74, 70, 75, 69, 76, 71, 73],
            },
            {
              variable: "heart_rate",
              stat_name: "sd" as const,
              observed_value: 1.8,
              rep_values: [2.0, 2.1, 1.9, 2.2, 2.5, 1.8, 1.7, 2.3, 2.0, 2.1],
            },
          ],
        },
        power_scaling: [],
      } as AllStageData["stage-5b"],
    };
    const result = generateMarkdown(data, "run-5-ppc");
    expect(result).toContain("Posterior Predictive Overlays");
    expect(result).toContain("heart_rate");
    expect(result).toContain("95% CI coverage");
    expect(result).toContain("4/4"); // 4 non-null, all within band
    expect(result).toContain("RMSE:");
    expect(result).toContain("MAE:");
    expect(result).toContain("Pearson r:");
    expect(result).toContain("Test Statistics");
    expect(result).toContain("mean");
    expect(result).toContain("sd");
    expect(result).toContain("72.500"); // observed value
  });

  it("includes stage 5b LOO-PIT and Pareto k", () => {
    const data: AllStageData = {
      "stage-5b": {
        outcome: "success",
        inference_metadata: { method: "nuts", n_samples: 1000, duration_seconds: 60 },
        loo_diagnostics: {
          elpd_loo: -120.5,
          p_loo: 3.2,
          se: 8.1,
          n_data_points: 100,
          observation_unit: "timestep",
          n_bad_k: 2,
          loo_pit: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
          pareto_k: [0.1, 0.2, 0.3, 0.8, 0.6, 0.4, 0.05, 0.9, 0.15, 0.25],
        },
        ppc: { per_variable_warnings: [], overlays: [], test_stats: [] },
        power_scaling: [],
      } as AllStageData["stage-5b"],
    };
    const result = generateMarkdown(data, "run-5-loo");
    expect(result).toContain("LOO Cross-Validation");
    expect(result).toContain("-120.5");
    expect(result).toContain("timestep");
    expect(result).toContain("LOO-PIT");
    expect(result).toContain("should be uniform");
    expect(result).toContain("KS stat:");
    expect(result).toContain("Calibration:");
    expect(result).toContain("Pareto k Diagnostics");
    expect(result).toContain("k > 0.7 (fail)");
    expect(result).toContain("2"); // 2 bad k (0.8 and 0.9)
  });

  it("includes stage 5b posterior pairs", () => {
    const data: AllStageData = {
      "stage-5b": {
        outcome: "success",
        inference_metadata: { method: "nuts", n_samples: 1000, duration_seconds: 60 },
        ppc: { per_variable_warnings: [], overlays: [], test_stats: [] },
        power_scaling: [],
        posterior_pairs: [
          {
            param_x: "rho_x",
            param_y: "sigma_x",
            x_values: [0.4, 0.5, 0.6, 0.45, 0.55],
            y_values: [0.1, 0.12, 0.08, 0.11, 0.09],
            divergent: [false, false, true, false, false],
          },
        ],
      } as AllStageData["stage-5b"],
    };
    const result = generateMarkdown(data, "run-5-pairs");
    expect(result).toContain("Posterior Pairs");
    expect(result).toContain("rho_x vs sigma_x");
    expect(result).toContain("1 divergent");
    expect(result).toContain("Pearson r:");
  });

  it("includes stage 6 prior sensitivity warnings and manifest effects", () => {
    const data: AllStageData = {
      "stage-6": {
        outcome: "success",
        intervention_results: [
          {
            treatment: "exercise",
            effect_size: 0.35,
            identifiable: true,
            prob_positive: 0.92,
            posterior_draws: [0.1, 0.2, 0.3, 0.4, 0.5],
            prior_sensitivity_warning: "Effect dominated by prior",
            manifest_effects: { daily_steps: 0.28, active_minutes: 0.15 },
          },
          {
            treatment: "diet",
            effect_size: -0.1,
            identifiable: false,
            prob_positive: 0.3,
            posterior_draws: [-0.2, -0.1, 0, 0.1, -0.15],
          },
        ],
      } as AllStageData["stage-6"],
    };
    const result = generateMarkdown(data, "run-6");
    expect(result).toContain("Status");
    expect(result).toContain("prior-sensitive");
    expect(result).toContain("non-identifiable");
    expect(result).toContain("Prior Sensitivity Warnings");
    expect(result).toContain("Effect dominated by prior");
    expect(result).toContain("Manifest Effects");
    expect(result).toContain("daily_steps");
    expect(result).toContain("active_minutes");
  });

  it("handles stage 5a with null optional fields", () => {
    const data: AllStageData = {
      "stage-5a": {
        outcome: "success",
        inference_metadata: { method: "svi", n_samples: 100, duration_seconds: 5.0 },
      } as AllStageData["stage-5a"],
    };
    const result = generateMarkdown(data, "run-5a-minimal");
    expect(result).toContain("Stage 5a");
    expect(result).toContain("SVI Preflight");
    expect(result).not.toContain("Posterior Marginals");
    expect(result).not.toContain("Posterior Pairs");
  });

  it("handles stage 5b with all diagnostic types simultaneously", () => {
    const data: AllStageData = {
      "stage-5b": {
        outcome: "success",
        inference_metadata: { method: "nuts", n_samples: 2000, duration_seconds: 120 },
        mcmc_diagnostics: {
          num_divergences: 3,
          divergence_rate: 0.0015,
          tree_depth_mean: 5.2,
          tree_depth_max: 10,
          accept_prob_mean: 0.85,
          num_chains: 4,
          num_samples: 2000,
          per_parameter: [
            { parameter: "rho_x", r_hat: 1.001, ess_bulk: 1800, ess_tail: 1600, mcse_mean: 0.002 },
          ],
          trace_data: [
            { parameter: "rho_x", chains: [{ chain: 0, values: [0.5, 0.6, 0.55, 0.52, 0.58] }] },
          ],
          rank_histograms: [
            {
              parameter: "rho_x",
              n_bins: 4,
              expected_per_bin: 125,
              chains: [{ chain: 0, counts: [120, 130, 125, 125] }],
            },
          ],
          energy: {
            bfmi: [0.9],
            energy_hist: { bin_centers: [1, 2, 3], density: [0.2, 0.6, 0.2] },
            energy_transition_hist: { bin_centers: [0.5, 1, 1.5], density: [0.3, 0.5, 0.2] },
          },
        },
        svi_diagnostics: { elbo_losses: [100, 50, 30, 25, 22] },
        ppc: {
          per_variable_warnings: [
            { variable: "hr", check_type: "calibration" as const, passed: true, value: 0.5, message: "OK" },
          ],
          overlays: [
            {
              variable: "hr",
              observed: [70, 72],
              q025: [65, 67],
              q25: [68, 70],
              median: [71, 73],
              q75: [74, 76],
              q975: [78, 80],
              spaghetti_draws: [],
            },
          ],
          test_stats: [
            { variable: "hr", stat_name: "mean" as const, observed_value: 71, rep_values: [70, 72, 71] },
          ],
        },
        loo_diagnostics: {
          elpd_loo: -50,
          p_loo: 2.1,
          se: 5.0,
          n_data_points: 50,
          observation_unit: "timestep",
          loo_pit: [0.1, 0.5, 0.9],
          pareto_k: [0.1, 0.3, 0.5],
        },
        power_scaling: [
          { parameter: "rho_x", diagnosis: "well_identified" as const, prior_sensitivity: 0.01, likelihood_sensitivity: 0.02 },
          { parameter: "sigma_x", diagnosis: "prior_dominated" as const, prior_sensitivity: 0.15, likelihood_sensitivity: 0.01 },
        ],
        posterior_marginals: [
          { parameter: "rho_x", x_values: [0, 0.5, 1], density: [0.1, 1.0, 0.1], mean: 0.5, sd: 0.1, hdi_3: 0.3, hdi_97: 0.7 },
        ],
        posterior_pairs: [
          { param_x: "rho_x", param_y: "sigma_x", x_values: [0.5, 0.6], y_values: [0.1, 0.12], divergent: [false, true] },
        ],
      } as AllStageData["stage-5b"],
    };
    const result = generateMarkdown(data, "run-5-full");

    // Verify all sections are present
    expect(result).toContain("MCMC Diagnostics");
    expect(result).toContain("Convergence");
    expect(result).toContain("MCSE");
    expect(result).toContain("Trace Plots");
    expect(result).toContain("Rank Histograms");
    expect(result).toContain("Energy Diagnostics");
    expect(result).toContain("ELBO");
    expect(result).toContain("Posterior Predictive Checks");
    expect(result).toContain("Posterior Predictive Overlays");
    expect(result).toContain("Test Statistics");
    expect(result).toContain("LOO Cross-Validation");
    expect(result).toContain("LOO-PIT");
    expect(result).toContain("Power Scaling");
    expect(result).toContain("Posterior Marginals");
    expect(result).toContain("Posterior Pairs");
    expect(result).toContain("1 divergent");
  });

  it("handles full pipeline with all stages", () => {
    const data: AllStageData = {
      "stage-0": {
        outcome: "success",
        n_records: 1000,
        n_columns: 10,
        date_range: { start: "2024-01-01", end: "2024-12-31" },
        sample: [{ timestamp: "2024-01-01", value: "42" }],
        column_descriptions: [{ name: "timestamp", dtype: "datetime", description: "Time" }],
      },
      "stage-1a": {
        outcome: "success",
        latent_model: {
          constructs: [
            { name: "exercise", description: "Ex", role: "exogenous", is_outcome: false, temporal_status: "time_varying" },
            { name: "bp", description: "BP", role: "endogenous", is_outcome: true, temporal_status: "time_varying" },
          ],
          edges: [{ cause: "exercise", effect: "bp", lagged: true, description: "Lowers BP" }],
        },
      } as AllStageData["stage-1a"],
      "stage-3": {
        outcome: "success",
        is_valid: true,
        indicators: {},
        dataset_issues: [],
      } as AllStageData["stage-3"],
      "stage-5a": {
        outcome: "success",
        inference_metadata: { method: "svi", n_samples: 100, duration_seconds: 5 },
        svi_diagnostics: { elbo_losses: [100, 50, 30] },
      } as AllStageData["stage-5a"],
      "stage-5b": {
        outcome: "success",
        inference_metadata: { method: "nuts", n_samples: 1000, duration_seconds: 60 },
        ppc: { per_variable_warnings: [], overlays: [], test_stats: [] },
        power_scaling: [],
      } as AllStageData["stage-5b"],
      "stage-6": {
        outcome: "success",
        intervention_results: [
          { treatment: "exercise", effect_size: 0.3, identifiable: true, prob_positive: 0.9 },
        ],
      } as AllStageData["stage-6"],
    };
    const result = generateMarkdown(data, "full-pipeline");
    expect(result).toContain("Stage 0");
    expect(result).toContain("Stage 1a");
    expect(result).toContain("Stage 3");
    expect(result).toContain("Stage 5a");
    expect(result).toContain("Stage 5b");
    expect(result).toContain("Stage 6");
  });
});
