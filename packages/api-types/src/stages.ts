export const STAGE_IDS = [
  "stage-0",
  "stage-1a",
  "stage-1b",
  "stage-2",
  "stage-3",
  "stage-4",
  "stage-4b",
  "stage-5",
  "stage-6",
] as const;

export type StageId = (typeof STAGE_IDS)[number];

export interface StageMeta {
  id: StageId;
  label: string;
  number: string;
  hasGate: boolean;
  prefectFlowName: string;
  /** Human-readable hint shown while this stage is running. */
  loadingHint: string;
  /** Static subtitle describing what this stage does (dataset-agnostic). */
  description: string;
}

export const STAGES: StageMeta[] = [
  {
    id: "stage-0",
    label: "Preprocess",
    number: "0",
    hasGate: false,
    prefectFlowName: "stage-0-flow",
    loadingHint: "Parsing and preprocessing your data...",
    description: "Parses raw data files and prepares them for downstream analysis.",
  },
  {
    id: "stage-1a",
    label: "Latent Model",
    number: "1a",
    hasGate: false,
    prefectFlowName: "stage-1a-flow",
    loadingHint: "LLM is proposing a causal DAG...",
    description: "Proposes a latent causal model based on domain knowledge alone, specifying theoretical constructs and their causal relationships.",
  },
  {
    id: "stage-1b",
    label: "Measurement & Nonparametric Identification",
    number: "1b",
    hasGate: true,
    prefectFlowName: "stage-1b-flow",
    loadingHint: "Mapping indicators and checking identifiability...",
    description: "Maps latent constructs to observable indicators and verifies nonparametric identifiability via do-calculus.",
  },
  {
    id: "stage-2",
    label: "Data Extraction",
    number: "2",
    hasGate: false,
    prefectFlowName: "stage-2-flow",
    loadingHint: "Extracting indicator values from your data...",
    description: "Dispatches worker LLMs to extract indicator observations from raw activity data, processing each chunk independently.",
  },
  {
    id: "stage-3",
    label: "Validation",
    number: "3",
    hasGate: false,
    prefectFlowName: "stage-3-flow",
    loadingHint: "Validating extraction quality...",
    description: "Validates extraction quality, checking for missing data, outliers, and consistency across indicators.",
  },
  {
    id: "stage-4",
    label: "Model Specification",
    number: "4",
    hasGate: false,
    prefectFlowName: "stage-4-flow",
    loadingHint: "LLM is specifying priors and model parameters...",
    description: "Specifies prior distributions and model parameters using domain knowledge and empirical data.",
  },
  {
    id: "stage-4b",
    label: "Parametric Identifiability",
    number: "4b",
    hasGate: true,
    prefectFlowName: "stage-4b-flow",
    loadingHint: "Checking parametric identifiability...",
    description: "Checks whether the specified model parameters are identifiable from the available data.",
  },
  {
    id: "stage-5",
    label: "Inference & Diagnostics",
    number: "5",
    hasGate: false,
    prefectFlowName: "stage-5-flow",
    loadingHint: "Running Bayesian inference...",
    description: "Fits the Bayesian model via MCMC or SVI and runs convergence and sensitivity diagnostics.",
  },
  {
    id: "stage-6",
    label: "Treatment Effects",
    number: "6",
    hasGate: false,
    prefectFlowName: "stage-6-flow",
    loadingHint: "Computing interventional effects...",
    description: "Computes interventional treatment effects and ranks them by magnitude and certainty.",
  },
];
