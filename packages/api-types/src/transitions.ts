export const ARTIFACT_IDS = [
  "question",
  "raw_data",
  "latent_structure",
  "measurement_structure",
  "causal_design",
  "structural_plan",
  "identification_report",
  "measurements",
  "panel",
  "validation_report",
  "statistical_model_spec",
  "compiled_ssm",
  "posterior",
  "baseline_report",
  "saved_scenarios",
] as const;

export type ArtifactId = (typeof ARTIFACT_IDS)[number];

export const ARTIFACT_VIEW_IDS = [
  "raw_data",
  "latent_structure",
  "measurement_structure",
  "measurements",
  "validation_report",
  "statistical_model_spec",
  "posterior",
  "baseline_report",
] as const;

export type ArtifactViewId = (typeof ARTIFACT_VIEW_IDS)[number];

export type TransitionId = Exclude<ArtifactViewId, "validation_report">;

export type TransitionLogScopePolicy = "subflow" | "subflow-with-children";

export interface TransitionMeta {
  id: ArtifactViewId;
  label: string;
  /** Human-readable hint shown while this transition is running. */
  loadingHint: string;
  /** Static subtitle describing what this artifact view represents. */
  description: string;
  /** Whether the LLM trace panel allows refinement (chat input + apply). */
  interactive: boolean;
  /** How transition logs should expand from the subflow at runtime. */
  logScopePolicy?: TransitionLogScopePolicy;
}

export const TRANSITIONS: TransitionMeta[] = [
  {
    id: "raw_data",
    label: "Preprocess",
    loadingHint: "Parsing and preprocessing your data...",
    description: "Parses raw data files and prepares them for downstream analysis.",
    interactive: true,
  },
  {
    id: "latent_structure",
    label: "Latent Structure",
    loadingHint: "LLM is proposing a causal DAG...",
    description:
      "Proposes a latent causal structure based on domain knowledge alone, specifying theoretical constructs and their causal relationships.",
    interactive: true,
  },
  {
    id: "measurement_structure",
    label: "Measurement Structure & Identification",
    loadingHint: "Mapping indicators and checking identifiability...",
    description:
      "Maps latent constructs to observable indicators and verifies nonparametric identifiability via do-calculus.",
    interactive: true,
  },
  {
    id: "measurements",
    label: "Data Extraction",
    loadingHint: "Extracting indicator values from your data...",
    description:
      "Dispatches worker LLMs to extract indicator observations from raw activity data, processing each chunk independently.",
    interactive: false,
    logScopePolicy: "subflow-with-children",
  },
  {
    id: "validation_report",
    label: "Validation",
    loadingHint: "Validating extraction quality...",
    description:
      "Validates extraction quality, checking for missing data, outliers, and consistency across indicators.",
    interactive: false,
  },
  {
    id: "statistical_model_spec",
    label: "Statistical Model Spec",
    loadingHint: "LLM is specifying the statistical model and priors...",
    description:
      "Specifies observation likelihoods, SSM parameters, and prior distributions using domain knowledge and empirical data.",
    interactive: true,
  },
  {
    id: "posterior",
    label: "Inference & Diagnostics",
    loadingHint: "Running Bayesian inference...",
    description: "Fits the Bayesian model and runs convergence and sensitivity diagnostics.",
    interactive: false,
  },
  {
    id: "baseline_report",
    label: "Treatment Effects",
    loadingHint: "Computing interventional effects...",
    description:
      "Computes interventional treatment effects and ranks them by magnitude and certainty.",
    interactive: true,
  },
];

export const TRANSITION_META: Record<ArtifactViewId, TransitionMeta> = Object.fromEntries(
  TRANSITIONS.map((transition) => [transition.id, transition]),
) as Record<ArtifactViewId, TransitionMeta>;
