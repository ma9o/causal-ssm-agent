/* eslint-disable */
/**
 * AUTO-GENERATED — DO NOT EDIT
 *
 * Generated from Python ToolContract definitions via:
 *   cd apps/data-pipeline && uv run python scripts/export_schemas.py
 *   cd packages/api-types && bun run scripts/generate.ts
 *
 * Source of truth: apps/data-pipeline/src/nof1_causal_lab/flows/artifact_contracts.py
 */

export interface ToolDefinition {
  name: string;
  description: string;
  /** JSON Schema for the tool's input parameters */
  parameters: Record<string, unknown>;
  /** JSON Schema for the tool's result payload, when declared */
  result?: Record<string, unknown> | null;
}

export const CONTEXT_TOOLS: Record<string, ToolDefinition[]> = {
  "ingestion": [
    {
      name: "list_files",
      description: "List files in the prepared input directory.",
      parameters: {"additionalProperties":false,"properties":{"path":{"default":".","description":"Relative path within the input directory.","title":"Path","type":"string"}},"title":"ListFilesInput","type":"object"},
      result: null,
    },
    {
      name: "read_file_sample",
      description: "Read a sample of lines from a file to understand its format.",
      parameters: {"additionalProperties":false,"properties":{"path":{"description":"Relative path to the file within the input directory.","title":"Path","type":"string"},"n_lines":{"default":50,"description":"Number of lines to read.","title":"N Lines","type":"integer"}},"required":["path"],"title":"ReadFileSampleInput","type":"object"},
      result: null,
    },
    {
      name: "execute_python",
      description: "Execute Python code in a Modal sandbox to parse files into a Polars DataFrame.",
      parameters: {"additionalProperties":false,"properties":{"code":{"description":"Python code to execute.","title":"Code","type":"string"}},"required":["code"],"title":"ExecutePythonInput","type":"object"},
      result: null,
    },
    {
      name: "submit_table",
      description: "Validate and finalize the ingested DataFrame with column descriptions.",
      parameters: {"additionalProperties":false,"properties":{"column_descriptions_json":{"description":"JSON object mapping column names to descriptions.","title":"Column Descriptions Json","type":"string"}},"required":["column_descriptions_json"],"title":"SubmitTableInput","type":"object"},
      result: null,
    },
  ],
  "latent-structure": [
    {
      name: "validate_latent_structure",
      description: "Tool for validating latent structure JSON (latent-structure).",
      parameters: {"additionalProperties":false,"properties":{"structure_json":{"description":"The JSON string containing the latent structure to validate.","title":"Structure Json","type":"string"}},"required":["structure_json"],"title":"ValidateLatentStructureInput","type":"object"},
      result: null,
    },
  ],
  "measurement-structure": [
    {
      name: "validate_measurement_structure",
      description: "Validate measurement structure JSON and compiler constraints.",
      parameters: {"additionalProperties":false,"properties":{"measurement_json":{"description":"The JSON string containing the measurement structure to validate.","title":"Measurement Json","type":"string"}},"required":["measurement_json"],"title":"ValidateMeasurementStructureInput","type":"object"},
      result: null,
    },
  ],
  "measurement": [
    {
      name: "validate_extractions",
      description: "Tool for validating worker extraction output JSON.",
      parameters: {"additionalProperties":false,"properties":{"output_json":{"description":"The JSON string containing the worker output to validate.","title":"Output Json","type":"string"}},"required":["output_json"],"title":"ValidateExtractionsInput","type":"object"},
      result: null,
    },
  ],
  "statistical-model-spec": [
    {
      name: "search_literature",
      description: "Search for empirical literature about effect sizes for model parameters.",
      parameters: {"additionalProperties":false,"properties":{"query":{"description":"Search query for empirical literature about effect sizes.","title":"Query","type":"string"},"parameter_name":{"description":"Name of the parameter this search is for (e.g. 'beta_stress_sleep').","title":"Parameter Name","type":"string"}},"required":["query","parameter_name"],"title":"SearchLiteratureInput","type":"object"},
      result: null,
    },
    {
      name: "submit_statistical_model_spec",
      description: "Submit the full model-spec StatisticalModelSpec for compile-only locking and validation.",
      parameters: {"additionalProperties":false,"properties":{"statistical_model_spec_json":{"description":"The JSON string containing the complete StatisticalModelSpec to lock for model-spec.","title":"Statistical Model Spec Json","type":"string"}},"required":["statistical_model_spec_json"],"title":"SubmitStatisticalModelSpecInput","type":"object"},
      result: null,
    },
    {
      name: "submit_priors",
      description: "Submit model-spec prior proposals for schema, compile, and prior-predictive validation.",
      parameters: {"additionalProperties":false,"properties":{"priors_json":{"description":"The JSON string containing prior proposals keyed by parameter name.","title":"Priors Json","type":"string"}},"required":["priors_json"],"title":"SubmitPriorsInput","type":"object"},
      result: null,
    },
  ],
  "ranking": [
    {
      name: "get_model_info",
      description: "Return a read-only summary of the fitted model, variables, identifiability status, diagnostics, and baseline effects.",
      parameters: {"additionalProperties":false,"properties":{"sections":{"description":"Named sections to include in the read-only model summary.","items":{"enum":["overview","variables","measurement","identifiability","diagnostics","baseline_effects","capabilities"],"type":"string"},"title":"Sections","type":"array"},"names":{"description":"Optional construct or indicator names to focus the summary on.","items":{"type":"string"},"title":"Names","type":"array"}},"title":"GetModelInfoInput","type":"object"},
      result: null,
    },
    {
      name: "simulate",
      description: "Run a composable causal scenario on the fitted generative model. Start from the population baseline steady state (interventional) or an abducted fitted latent state (counterfactual), apply one or more timed latent clamps (do-operators), and read the effect on an outcome over a horizon.",
      parameters: {"additionalProperties":false,"properties":{"start":{"additionalProperties":false,"description":"Where the forward rollout begins (replaces the rung-2/rung-3 split).","properties":{"kind":{"default":"baseline","description":"'baseline' starts from the population baseline steady state (an interventional, rung-2 query). 'abducted' conditions on the individual's observed evidence and starts from the recovered fitted latent state (a counterfactual, rung-3 query).","enum":["baseline","abducted"],"title":"Kind","type":"string"},"time_index":{"anyOf":[{"minimum":0,"type":"integer"},{"type":"null"}],"default":null,"description":"Abducted start only: observed fitted-state index to begin from. Defaults to the final retained fitted latent state.","title":"Time Index"},"time":{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"description":"Abducted start only: ISO-8601 observed timestamp matching a retained fitted latent state. Use either time_index or time, not both.","title":"Time"}},"title":"ScenarioStartInput","type":"object"},"clamps":{"description":"One or more timed latent clamps composing the scenario.","items":{"additionalProperties":false,"description":"A do-operator on one latent variable over a time window.\n\nThe window is ``[from_day, to_day)`` in days relative to the rollout start; outside\nthe window the variable evolves under its natural dynamics. ``set`` pins to an absolute\nvalue, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly\ninterpolates across the window, and ``trajectory`` tracks a list of values across it.","properties":{"variable":{"description":"Latent construct to clamp.","title":"Variable","type":"string"},"mode":{"description":"How the clamped value is specified over the window.","enum":["set","shift","ramp","trajectory"],"title":"Mode","type":"string"},"value":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Required when mode='set'. Absolute latent-space value.","title":"Value"},"amount":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Required when mode='shift'. Additive delta from the start-state value.","title":"Amount"},"value_start":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Required when mode='ramp'. Value at from_day.","title":"Value Start"},"value_end":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Required when mode='ramp'. Value at to_day.","title":"Value End"},"values":{"anyOf":[{"items":{"type":"number"},"type":"array"},{"type":"null"}],"default":null,"description":"Required when mode='trajectory'. Values sampled evenly across the window.","title":"Values"},"from_day":{"default":0,"description":"Window onset in days from the rollout start.","minimum":0,"title":"From Day","type":"number"},"to_day":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Window end in days from the rollout start. Null runs through the horizon.","title":"To Day"}},"required":["variable","mode"],"title":"LatentClampInput","type":"object"},"minItems":1,"title":"Clamps","type":"array"},"outcome":{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"description":"Outcome construct. Defaults to the latent-structure outcome.","title":"Outcome"},"query":{"additionalProperties":false,"properties":{"estimand":{"default":"trajectory","description":"Report the final-horizon outcome effect or the full effect trajectory.","enum":["end_state","trajectory"],"title":"Estimand","type":"string"},"horizon_days":{"default":30,"description":"Forward horizon in days from the rollout start.","maximum":365,"minimum":1,"title":"Horizon Days","type":"integer"},"projection":{"default":"latent","description":"Report latent outcome effects, manifest projections, or both.","enum":["latent","manifest","both"],"title":"Projection","type":"string"}},"title":"ScenarioQueryInput","type":"object"}},"required":["clamps"],"title":"SimulateScenarioInput","type":"object"},
      result: {"anyOf":[{"additionalProperties":false,"properties":{"start":{"additionalProperties":false,"properties":{"kind":{"enum":["baseline","abducted"],"title":"Kind","type":"string"},"time_index":{"anyOf":[{"type":"integer"},{"type":"null"}],"default":null,"title":"Time Index"},"time":{"anyOf":[{"type":"string"},{"type":"null"}],"default":null,"title":"Time"},"state_source":{"enum":["baseline_steady_state","fitted_latent_paths"],"title":"State Source","type":"string"}},"required":["kind","state_source"],"title":"ScenarioStartResultContract","type":"object"},"clamps":{"items":{"additionalProperties":false,"description":"A do-operator on one latent variable over a time window.\n\nThe window is ``[from_day, to_day)`` in days relative to the rollout start; outside\nthe window the variable evolves under its natural dynamics. ``set`` pins to an absolute\nvalue, ``shift`` adds an amount to the variable's start-state value, ``ramp`` linearly\ninterpolates across the window, and ``trajectory`` tracks a list of values across it.","properties":{"variable":{"description":"Latent construct to clamp.","title":"Variable","type":"string"},"mode":{"description":"How the clamped value is specified over the window.","enum":["set","shift","ramp","trajectory"],"title":"Mode","type":"string"},"value":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Required when mode='set'. Absolute latent-space value.","title":"Value"},"amount":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Required when mode='shift'. Additive delta from the start-state value.","title":"Amount"},"value_start":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Required when mode='ramp'. Value at from_day.","title":"Value Start"},"value_end":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Required when mode='ramp'. Value at to_day.","title":"Value End"},"values":{"anyOf":[{"items":{"type":"number"},"type":"array"},{"type":"null"}],"default":null,"description":"Required when mode='trajectory'. Values sampled evenly across the window.","title":"Values"},"from_day":{"default":0,"description":"Window onset in days from the rollout start.","minimum":0,"title":"From Day","type":"number"},"to_day":{"anyOf":[{"type":"number"},{"type":"null"}],"default":null,"description":"Window end in days from the rollout start. Null runs through the horizon.","title":"To Day"}},"required":["variable","mode"],"title":"LatentClampInput","type":"object"},"title":"Clamps","type":"array"},"outcome":{"title":"Outcome","type":"string"},"estimand":{"enum":["end_state","trajectory"],"title":"Estimand","type":"string"},"summary":{"additionalProperties":false,"properties":{"mean":{"title":"Mean","type":"number"},"median":{"title":"Median","type":"number"},"lower_95":{"title":"Lower 95","type":"number"},"upper_95":{"title":"Upper 95","type":"number"},"prob_positive":{"title":"Prob Positive","type":"number"}},"required":["mean","median","lower_95","upper_95","prob_positive"],"title":"EffectSummaryContract","type":"object"},"effect_trajectory":{"anyOf":[{"items":{"additionalProperties":false,"properties":{"day":{"title":"Day","type":"number"},"effect":{"title":"Effect","type":"number"}},"required":["day","effect"],"title":"EffectTrajectoryPointContract","type":"object"},"type":"array"},{"type":"null"}],"default":null,"title":"Effect Trajectory"},"visualization":{"anyOf":[{"additionalProperties":false,"properties":{"reference_node_trajectories":{"anyOf":[{"additionalProperties":{"items":{"type":"number"},"type":"array"},"type":"object"},{"type":"null"}],"default":null,"description":"Per-construct latent trajectories for the reference (no-clamp) path aligned to effect_trajectory days.","title":"Reference Node Trajectories"},"action_node_trajectories":{"anyOf":[{"additionalProperties":{"items":{"type":"number"},"type":"array"},"type":"object"},{"type":"null"}],"default":null,"description":"Per-construct latent trajectories under the composed clamps aligned to effect_trajectory days.","title":"Action Node Trajectories"},"node_effect_trajectories":{"anyOf":[{"additionalProperties":{"items":{"type":"number"},"type":"array"},"type":"object"},{"type":"null"}],"default":null,"description":"Per-construct latent effect trajectories aligned to effect_trajectory days. Values are causal deltas relative to the reference path.","title":"Node Effect Trajectories"},"start_state":{"anyOf":[{"additionalProperties":{"type":"number"},"type":"object"},{"type":"null"}],"default":null,"description":"Posterior mean latent state the rollout started from.","title":"Start State"}},"title":"BaselineReportVisualizationContract","type":"object"},{"type":"null"}],"default":null},"manifest_effects":{"anyOf":[{"additionalProperties":{"type":"number"},"type":"object"},{"type":"null"}],"default":null,"title":"Manifest Effects"},"reference_mean":{"description":"Mean reference outcome (baseline steady state or factual forecast).","title":"Reference Mean","type":"number"},"warnings":{"items":{"type":"string"},"title":"Warnings","type":"array"}},"required":["start","clamps","outcome","estimand","summary","reference_mean"],"title":"SimulateScenarioResultContract","type":"object"},{"additionalProperties":false,"properties":{"error":{"title":"Error","type":"string"},"identifiable_treatments":{"anyOf":[{"items":{"type":"string"},"type":"array"},{"type":"null"}],"default":null,"title":"Identifiable Treatments"}},"required":["error"],"title":"ToolErrorContract","type":"object"}],"title":"SimulateScenarioToolResultContract"},
    },
  ],
};

export const INTERACTIVE_CONTEXTS: readonly string[] = ["latent-structure","measurement-structure","ranking","statistical-model-spec"] as const;
