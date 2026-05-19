export const STAGE_IDS = [
  "stage-0",
  "stage-1a",
  "stage-1b",
  "stage-2",
  "stage-3",
  "stage-4",
  "stage-4b",
  "stage-5b",
  "stage-6",
] as const;

export type StageId = (typeof STAGE_IDS)[number];

export type StageLogScopePolicy = "subflow" | "subflow-with-children";

export interface StageMeta {
  id: StageId;
  label: string;
  number: string;
  /** Human-readable hint shown while this stage is running. */
  loadingHint: string;
  /** Static subtitle describing what this stage does (dataset-agnostic). */
  description: string;
  /** Whether the LLM trace panel allows refinement (chat input + apply). */
  interactive: boolean;
  /** How stage logs should expand from the stage subflow at runtime. */
  logScopePolicy?: StageLogScopePolicy;
}

export const STAGES: StageMeta[] = [
  {
    id: "stage-0",
    label: "Preprocess",
    number: "0",
    loadingHint: "Parsing and preprocessing your data...",
    description: "Parses raw data files and prepares them for downstream analysis.",
    interactive: true,
  },
  {
    id: "stage-1a",
    label: "Latent Model",
    number: "1a",
    loadingHint: "LLM is proposing a causal DAG...",
    description:
      "Proposes a latent causal model based on domain knowledge alone, specifying theoretical constructs and their causal relationships.",
    interactive: true,
  },
  {
    id: "stage-1b",
    label: "Measurement & Nonparametric Identification",
    number: "1b",
    loadingHint: "Mapping indicators and checking identifiability...",
    description:
      "Maps latent constructs to observable indicators and verifies nonparametric identifiability via do-calculus.",
    interactive: true,
  },
  {
    id: "stage-2",
    label: "Data Extraction",
    number: "2",
    loadingHint: "Extracting indicator values from your data...",
    description:
      "Dispatches worker LLMs to extract indicator observations from raw activity data, processing each chunk independently.",
    interactive: false,
    logScopePolicy: "subflow-with-children",
  },
  {
    id: "stage-3",
    label: "Validation",
    number: "3",
    loadingHint: "Validating extraction quality...",
    description:
      "Validates extraction quality, checking for missing data, outliers, and consistency across indicators.",
    interactive: false,
  },
  {
    id: "stage-4",
    label: "Model Specification",
    number: "4",
    loadingHint: "LLM is specifying priors and model parameters...",
    description:
      "Specifies prior distributions and model parameters using domain knowledge and empirical data.",
    interactive: true,
  },
  {
    id: "stage-4b",
    label: "Parametric Identifiability",
    number: "4b",
    loadingHint: "Checking parametric identifiability...",
    description:
      "Checks whether the specified model parameters are identifiable from the available data.",
    interactive: false,
  },
  {
    id: "stage-5b",
    label: "Inference & Diagnostics",
    number: "5b",
    loadingHint: "Running Bayesian inference...",
    description:
      "Fits the Bayesian model and runs convergence and sensitivity diagnostics.",
    interactive: false,
  },
  {
    id: "stage-6",
    label: "Treatment Effects",
    number: "6",
    loadingHint: "Computing interventional effects...",
    description:
      "Computes interventional treatment effects and ranks them by magnitude and certainty.",
    interactive: true,
  },
];
