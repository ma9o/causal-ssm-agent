import type { Meta, StoryObj } from "@storybook/nextjs-vite";
import type { Stage4bData } from "@nof1-causal-lab/api-types";
import { withContainer } from "@/components/story-decorators";
import { MapGeometryPanel } from "./map-geometry-panel";

type MAPGeometryResult = NonNullable<Stage4bData["parametric_id"]["map_geometry"]>;

const sampleResult: MAPGeometryResult = {
  n_starts: 4,
  n_successful_starts: 3,
  best_start_index: 1,
  map_log_posterior: -118.4321,
  map_log_likelihood: -101.2284,
  map_log_prior: -17.2037,
  final_grad_norm: 0.00042,
  runner_up_objective_gap: 0.183,
  starts: [
    {
      index: 0,
      start_kind: "zero",
      start_log_posterior: -140.2,
      log_posterior: -118.73,
      log_likelihood: -101.49,
      log_prior: -17.24,
      objective: 118.73,
      success: true,
      status: 0,
      message: "ok",
      n_iters: 32,
      n_function_evals: 44,
      grad_norm: 0.0018,
      distance_to_best: 0.23,
    },
    {
      index: 1,
      start_kind: "prior_median",
      start_log_posterior: -126.8,
      log_posterior: -118.4321,
      log_likelihood: -101.2284,
      log_prior: -17.2037,
      objective: 118.4321,
      success: true,
      status: 0,
      message: "ok",
      n_iters: 21,
      n_function_evals: 30,
      grad_norm: 0.00042,
      distance_to_best: 0,
    },
    {
      index: 2,
      start_kind: "prior_draw_0",
      start_log_posterior: -130.1,
      log_posterior: -118.61,
      log_likelihood: -101.33,
      log_prior: -17.28,
      objective: 118.61,
      success: true,
      status: 0,
      message: "ok",
      n_iters: 28,
      n_function_evals: 39,
      grad_norm: 0.00091,
      distance_to_best: 0.11,
    },
    {
      index: 3,
      start_kind: "prior_draw_1",
      start_log_posterior: -154.4,
      log_posterior: -132.04,
      log_likelihood: -112.84,
      log_prior: -19.2,
      objective: 132.04,
      success: false,
      status: 1,
      message: "maxiter reached",
      n_iters: 50,
      n_function_evals: 68,
      grad_norm: 0.14,
      distance_to_best: 2.9,
    },
  ],
  likelihood_curvature: {
    eigenvalues: [220.4, 43.6, 8.9, 0.92, 0.34],
    normalized_eigenvalues: [41.8, 12.6, 4.3, 0.84, 0.18],
    negative_direction_count: 0,
    deficiency_count: 2,
    positive_definite: true,
    condition_number: 648.2,
    normalized_condition_number: 232.2,
    weak_directions: [
      {
        index: 4,
        eigenvalue: 0.92,
        normalized_eigenvalue: 0.84,
        status: "fail",
        top_loadings: [
          {
            parameter: "lambda_free[0]",
            interpretable_parameter: "lambda_sleep_problem",
            loading: 0.81,
            abs_loading: 0.81,
          },
          {
            parameter: "drift_offdiag_free[0]",
            interpretable_parameter: "beta_fatigue_sleep",
            loading: -0.41,
            abs_loading: 0.41,
          },
        ],
      },
      {
        index: 5,
        eigenvalue: 0.34,
        normalized_eigenvalue: 0.18,
        status: "fail",
        top_loadings: [
          {
            parameter: "manifest_var_diag_free[1]",
            interpretable_parameter: "obs_sd_fatigue",
            loading: 0.88,
            abs_loading: 0.88,
          },
          {
            parameter: "lambda_free[1]",
            interpretable_parameter: "lambda_sleepiness",
            loading: 0.29,
            abs_loading: 0.29,
          },
        ],
      },
    ],
    per_parameter: [
      {
        parameter: "lambda_free[0]",
        interpretable_parameter: "lambda_sleep_problem",
        diagonal_curvature: 2.4,
        effective_eigenvalue: 0.92,
        status: "warn",
        normalized_effective_eigenvalue: 0.84,
        normalized_status: "fail",
      },
      {
        parameter: "drift_offdiag_free[0]",
        interpretable_parameter: "beta_fatigue_sleep",
        diagonal_curvature: 8.7,
        effective_eigenvalue: 0.92,
        status: "warn",
        normalized_effective_eigenvalue: 4.3,
        normalized_status: "warn",
      },
      {
        parameter: "manifest_var_diag_free[1]",
        interpretable_parameter: "obs_sd_fatigue",
        diagonal_curvature: 0.6,
        effective_eigenvalue: 0.34,
        status: "fail",
        normalized_effective_eigenvalue: 0.18,
        normalized_status: "fail",
      },
    ],
  },
  posterior_curvature: {
    eigenvalues: [245.8, 57.1, 13.2, 4.2, 1.6],
    normalized_eigenvalues: [46.7, 17.4, 7.3, 3.1, 1.2],
    negative_direction_count: 0,
    deficiency_count: 0,
    positive_definite: true,
    condition_number: 153.6,
    normalized_condition_number: 38.9,
    weak_directions: [
      {
        index: 5,
        eigenvalue: 1.6,
        normalized_eigenvalue: 1.2,
        status: "warn",
        top_loadings: [
          {
            parameter: "manifest_var_diag_free[1]",
            interpretable_parameter: "obs_sd_fatigue",
            loading: 0.74,
            abs_loading: 0.74,
          },
          {
            parameter: "lambda_free[1]",
            interpretable_parameter: "lambda_sleepiness",
            loading: 0.43,
            abs_loading: 0.43,
          },
        ],
      },
    ],
    per_parameter: [
      {
        parameter: "lambda_free[0]",
        interpretable_parameter: "lambda_sleep_problem",
        diagonal_curvature: 4.6,
        effective_eigenvalue: 4.2,
        status: "pass",
        normalized_effective_eigenvalue: 3.1,
        normalized_status: "warn",
      },
      {
        parameter: "drift_offdiag_free[0]",
        interpretable_parameter: "beta_fatigue_sleep",
        diagonal_curvature: 13.1,
        effective_eigenvalue: 7.3,
        status: "pass",
        normalized_effective_eigenvalue: 7.3,
        normalized_status: "warn",
      },
      {
        parameter: "manifest_var_diag_free[1]",
        interpretable_parameter: "obs_sd_fatigue",
        diagonal_curvature: 1.8,
        effective_eigenvalue: 1.6,
        status: "warn",
        normalized_effective_eigenvalue: 1.2,
        normalized_status: "warn",
      },
    ],
  },
  prior_rescued_parameters: ["lambda_free[0]"],
  boundary_parameters: ["manifest_var_diag_free[1]"],
};

const meta = {
  title: "Stages/ParametricId/MapGeometryPanel",
  component: MapGeometryPanel,
  decorators: [withContainer("max-w-6xl")],
} satisfies Meta<typeof MapGeometryPanel>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: { result: sampleResult },
};
