/** MCMC and model checking diagnostic thresholds. */

// R-hat convergence thresholds (Vehtari et al., 2021)
export const RHAT_FAIL = 1.1;
export const RHAT_WARN = 1.01;

// ESS ratio thresholds (effective sample size / total draws)
export const ESS_RATIO_FAIL = 0.1;
export const ESS_RATIO_WARN = 0.5;
export const DEFAULT_N_SAMPLES = 1000;

// Pareto-k diagnostic thresholds (Vehtari, Simpson & Gelman, 2024)
export const PARETO_K_FAIL = 0.7;
export const PARETO_K_WARN = 0.5;

// Power-scaling sensitivity threshold
export const POWER_SCALING_THRESHOLD = 0.05;

// PPC p-value tail thresholds
export const PPC_P_LOWER = 0.05;
export const PPC_P_UPPER = 0.95;

// Credible interval quantiles
export const CI_LOWER = 0.025;
export const CI_UPPER = 0.975;
